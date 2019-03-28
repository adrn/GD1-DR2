functions{

	  matrix get_cov(real h, real w) {
			  matrix[2, 2] C;
				C[1, 1] = h*h;
				C[2, 1] = 0.;
				C[1, 2] = 0.;
				C[2, 2] = w*w;
				return C;
		}

}

data {
		// number of pixels in the density map
		int n_pix;
		// number counts of objects in pixels
		int hh[n_pix];
		// the x locations of pixels
		vector[n_pix] x;
		// the y locations of pixels
		vector[n_pix] y;

		// ----------------------------------------------------------
		// Nodes:

		// number of nodes
		int n_nodes;

		// nodes locations along rigid polynomial
		vector[n_nodes] phi1_nodes;
		vector[n_nodes] phi2_nodes_init;

		// width of nodes: along rigid polynomial, h
		vector[n_nodes] h_nodes;

}

parameters {
		vector[n_nodes] d_phi2_nodes;
		vector<lower=-3, upper=0.5>[n_nodes] log_w_nodes;
		vector[n_nodes] log_N_nodes;
}

transformed parameters {
		// vector[n_pix] logbg_pix;
		vector[n_pix] logint_pix;
		vector[n_pix] xmod;
		vector[n_nodes] tmp;
		vector[2] xy;
		vector[2] node_xy;

		vector[n_nodes] phi2_nodes;

		// Un-log some things
		vector[n_nodes] w_nodes;
		vector[n_nodes] N_nodes;

		w_nodes = exp(log_w_nodes);
		N_nodes = exp(log_N_nodes);

		phi2_nodes = phi2_nodes_init + d_phi2_nodes;

	  // log densities of the background/stream at each pixel
	  // logbg_pix = logbg_val + bgsl_val/10 .* y + bgsl2_val/100 .* y .* y;
	  // logint_pix = logint_val - 0.5 * square(y-fi2_val) ./ exp(2 * logwidth_val);
	  for (i in 1:n_pix) {
			  xy[1] = x[i];
				xy[2] = y[i];
		    for (j in 1:n_nodes) {
					  node_xy[1] = phi1_nodes[j];
						node_xy[2] = phi2_nodes[j];
					  // tmp[j] = log_N_nodes[j] + normal_lpdf(y[i] | phi2_nodes[j], w_nodes[j]);
						tmp[j] = log_N_nodes[j] + multi_normal_lpdf(xy | node_xy, get_cov(h_nodes[j], w_nodes[j]));
				}
		    xmod[i] = log_sum_exp(tmp);
	  }
}
model {
	// Priors
	target += normal_lpdf(d_phi2_nodes | 0, 0.5);
	// target += normal_lpdf(log_N_nodes | 0, 0.5);
	target += normal_lpdf(log_w_nodes | log(0.25), 0.5);

	//Likelihood
	hh ~ poisson_log(xmod);
}

// generated quantities {
// 	real log_lik;
// 	// Save the actual log-likelihood of the function evaluation
// 	log_lik = poisson_log_lpmf(hh | xmod);
// }
