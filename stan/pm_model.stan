functions {

    real get_predicted_pm(real phi1, int deg, vector coeff) {
        real val;
        val = 0.;
        for (i in 1:deg+1) {
            val += coeff[i] * pow(phi1, i-1);
        }
        return val;
    }

    // get the vector of spacings between nodes
    vector geths(int n_nodes, vector nodes)
    {
      int n = n_nodes -1;
      vector[n] hs;
      for (i in 1:n)
      {
        hs[i] = nodes[i+1]-nodes[i];
      }
      return hs;
    }
    // obtain the vector of spline coefficients given the location
    // of the nodes and values there
    // We are using natural spline definition
    vector getcoeffs(int n_nodes, vector nodes, vector vals)
    {
      int n=n_nodes-1;
      vector[n] hi;
      vector[n] bi;
      vector[n-1] vi;
      vector[n-1] ui;
      vector[n_nodes] ret;
      vector[n-1] zs;
      matrix[n-1,n-1] M = rep_matrix(0, n-1, n-1);

      n = n_nodes-1;

      for (i in 1:n)
      {
        hi[i] = nodes[i+1]-nodes[i];
        bi[i] =  1/hi[i]*(vals[i+1]-vals[i]);
      }
      for (i in 2:n)
      {
        vi[i-1] = 2*(hi[i-1]+hi[i]);
        ui[i-1] = 6*(bi[i] - bi[i-1]);
      }
      for (i in 1:n-1)
      {
        M[i,i] = vi[i];
      }
      for (i in 1:n-2)
      {
        M[i+1,i] = hi[i];
        M[i,i+1] = hi[i];
      }
      //print (M)
      zs = M \ ui ; //mdivide_left_spd(M, ui);
      ret[1]=0;
      ret[n_nodes] =0;
      ret[2:n_nodes-1]=zs;

      return ret;

    }

    // Evaluate the spline, given nodes, values at the nodes
    // spline coefficients, locations of evaluation points
    // and integer bin ids of each point
    vector spline_eval(int n_nodes, vector nodes,
           vector vals, vector zs,
           int n_dat, vector x, int[] i)
    {

      vector[n_nodes-1] h;
      vector[n_dat] ret;
      int i1[n_dat];
      for (ii in 1:n_dat)
      {
        i1[ii] = i[ii] + 1;
      }
      h = geths(n_nodes, nodes);

      ret = (
          zs[i1] ./ 6 ./ h[i] .* square(x-nodes[i]) .*(x-nodes[i])+
          zs[i]  ./ 6 ./ h[i] .* square(nodes[i1]-x) .* (nodes[i1]-x)+
          (vals[i1] ./ h[i] - h[i] .* zs[i1] ./ 6) .* (x-nodes[i])+
          (vals[i] ./ h[i] - h[i] .* zs[i] ./ 6) .* (nodes[i1]-x)
          );
      return ret;
    }

    // find in which node interval we should place each point of the vector
    int[] findpos(int n_nodes, vector nodes, int n_dat, vector x)
    {
      int ret[n_dat];
      for (i in 1:n_dat)
      {
        for (j in 1:n_nodes-1)
        {
          if ((x[i]>=nodes[j]) && (x[i]<nodes[j+1]))
          {
            ret[i] = j;
          }
        }
      }
      return ret;
    }
}

data {
    int D; // number of dimensions

    // Bins for background model:
    int M;
    vector[M+1] phi1_bins;

    // Background model: fixed from using XD on control fields
    int K_bg; // number of mixture components in background model
    vector[D] mu_bg[M, K_bg];
    vector[K_bg] a_bg[M];
    matrix[D, D] cov_bg[M, K_bg];

    // Data:
    int N; // number of data points
    vector[N] phi1; // phi1 data
    vector[D] pm[N]; // proper motion data
    matrix[D, D] cov_pm[N]; // data covariance (uncertainty)

    // Nodes for the proper motion spline:
    int n_pm_nodes; // number of nodes
    vector[n_pm_nodes] pm_nodes; // phi1 locations of proper motion nodes
    vector[n_pm_nodes] pm1_nodes_init;
    vector[n_pm_nodes] pm2_nodes_init;
}

transformed data {
    vector[N] log_p_bg_noalpha;
    vector[K_bg] tmp;
    int bin_m;
    int node_ids_pm[N] = findpos(n_pm_nodes, pm_nodes, N, phi1);

    for (n in 1:N) {
        // First, figure out which phi1 bin this data point is in:
        for (m in 1:M) {
            if ((phi1[n] >= phi1_bins[m]) && (phi1[n] < phi1_bins[m+1])) {
                bin_m = m;
                break;
            }
        }

        // Now evaluate and cache the probability of belonging to the background
        // mixture model from that bin:
        for (k in 1:K_bg){
            tmp[k] = log(a_bg[bin_m, k]) + multi_normal_lpdf(pm[n] | mu_bg[bin_m, k], cov_bg[bin_m, k] + cov_pm[n]);
        }
        log_p_bg_noalpha[n] = log_sum_exp(tmp);
    }
}

parameters {
    // real<lower=0, upper=1> alpha; // mixing proportion
    // vector[D] mu_gd1; // GD-1 mixture component mean
    // real<lower=0, upper=1> s1_gd1; // GD-1 mixture component root-variance in phi1
    // real<lower=0, upper=1> s2_gd1; // GD-1 mixture component root-variance in phi2

    vector[n_pm_nodes] pm1_nodes;
    vector[n_pm_nodes] pm2_nodes;

    vector[n_pm_nodes] log_s1_nodes;
    vector[n_pm_nodes] log_s2_nodes;

    vector<lower=-15, upper=0>[n_pm_nodes] log_a_nodes;
}

transformed parameters {
    vector[N] log_p_stream;
    vector[N] log_p_bg;
    vector[D] predicted_pm;

    vector[N] spline_pm1;
    vector[N] spline_pm2;
    vector[N] spline_log_s1;
    vector[N] spline_log_s2;
    vector[N] spline_log_a;

    matrix[D, D] cov_gd1;

    vector[n_pm_nodes] coeffs_pm1 = getcoeffs(n_pm_nodes, pm_nodes, pm1_nodes);
    vector[n_pm_nodes] coeffs_pm2 = getcoeffs(n_pm_nodes, pm_nodes, pm2_nodes);
    vector[n_pm_nodes] coeffs_s1 = getcoeffs(n_pm_nodes, pm_nodes, log_s1_nodes);
    vector[n_pm_nodes] coeffs_s2 = getcoeffs(n_pm_nodes, pm_nodes, log_s2_nodes);
    vector[n_pm_nodes] coeffs_a = getcoeffs(n_pm_nodes, pm_nodes, log_a_nodes);

    // Evaluate spline at the location of each data point phi1
  	spline_pm1 = spline_eval(n_pm_nodes, pm_nodes, pm1_nodes, coeffs_pm1,
  	                         N, phi1, node_ids_pm);
    spline_pm2 = spline_eval(n_pm_nodes, pm_nodes, pm2_nodes, coeffs_pm2,
  	                         N, phi1, node_ids_pm);

    spline_log_s1 = spline_eval(n_pm_nodes, pm_nodes, log_s1_nodes, coeffs_s1,
  	                            N, phi1, node_ids_pm);
    spline_log_s2 = spline_eval(n_pm_nodes, pm_nodes, log_s2_nodes, coeffs_s2,
  	                            N, phi1, node_ids_pm);

    spline_log_a = spline_eval(n_pm_nodes, pm_nodes, log_a_nodes, coeffs_a,
  	                           N, phi1, node_ids_pm);

    cov_gd1[1, 2] = 0.;
    cov_gd1[2, 1] = 0.;

    for (n in 1:N) {
        predicted_pm[1] = spline_pm1[n];
        predicted_pm[2] = spline_pm2[n];

        cov_gd1[1, 1] = exp(2 * spline_log_s1[n]);
        cov_gd1[2, 2] = exp(2 * spline_log_s2[n]);

        log_p_stream[n] = spline_log_a[n] + multi_normal_lpdf(pm[n] | predicted_pm, cov_gd1 + cov_pm[n]);
        log_p_bg[n] = log_diff_exp(0, spline_log_a[n]) + log_p_bg_noalpha[n];
    }

}

model {
    // priors
    // s1_gd1 ~ normal(0, 1)T[0, 1];
    // s2_gd1 ~ normal(0, 1)T[0, 1];
    // alpha ~ normal(0.1, 0.05)T[0, 1];

    target += normal_lpdf(pm1_nodes | pm1_nodes_init, 0.5);
    target += normal_lpdf(pm2_nodes | pm2_nodes_init, 0.5);

    target += normal_lpdf(log_s1_nodes | log(0.1), 1);
    target += normal_lpdf(log_s2_nodes | log(0.1), 1);

    // target += normal_lpdf(log_a_nodes | log(0.1), 1);

    for (n in 1:N) {
        target += log_sum_exp(log_p_stream[n], log_p_bg[n]);
    }
}
