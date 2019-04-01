functions{

	  matrix get_cov(real h, real w) {
			  matrix[2, 2] C;
				C[1, 1] = square(h);
				C[2, 1] = 0.;
				C[1, 2] = 0.;
				C[2, 2] = square(w);
				return C;
		}

		matrix get_R(int i, int N, vector x, vector y) {
        matrix[2,2] R;
        vector[2] dxy;
        real theta;

        if (i == 1) {
            dxy[1] = (x[i+1] - x[i]);
            dxy[2] = (y[i+1] - y[i]);
        } else if (i == N) {
            dxy[1] = (x[i] - x[i-1]);
            dxy[2] = (y[i] - y[i-1]);
        } else {
            dxy[1] = (x[i+1] - x[i-1]);
            dxy[2] = (y[i+1] - y[i-1]);
        }
        dxy /= sqrt(sum(square(dxy)));
        theta = atan2(dxy[2], dxy[1]);

        R[1, 1] = cos(theta);
        R[2, 2] = R[1, 1];
        R[1, 2] = -sin(theta);
        R[2, 1] = sin(theta);

        return R;
		}

		real ln_bg_quadratic_uniform(real x1, real x2,
														     real a1, real b1, real a2, real b2,
															   real c1, real c2, real c3, real x0) {
				real lnA;
				real ln_px1;
				real ln_px2;
				real a;
				real b;

				a = a1;
				b = b1;
    		lnA = log(6) - log((b - a)*(2*a*a*c1 + 2*a*b*c1 + 2*b*b*c1 + 3*a*c2 + 3*b*c2 + 6*c3 - 6*a*c1*x0 - 6*b*c1*x0 - 6*c2*x0 + 6*c1*x0*x0));

				ln_px1 = lnA + log(c1*square(x1-x0) + c2*(x1-x0) + c3);

    		// x2 direction:
    		ln_px2 = -log(b2 - a2);

    		return ln_px1 + ln_px2;
		}
}

data {
		// number of pixels in the density map
		int n_pix;
		// number counts of objects in pixels
		int hh[n_pix];

		// selection function for each pixel
		real S[n_pix];

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

		// Window boundaries:
		real a1;
		real b1;
		real a2;
		real b2;

}

transformed data {
		real log_S[n_pix] = log(S); // selection function
    matrix[2,2] R[n_nodes]; // TODO: audit this

    // pre-compute rotation matrices
    for (i in 1:n_nodes) {
        R[i] = get_R(i, n_nodes, phi1_nodes, phi2_nodes_init);
    }
}

parameters {
		vector<lower=-0.75, upper=0.75>[n_nodes] d_phi2_nodes;
		vector<lower=-2, upper=-0.5>[n_nodes] log_w_nodes;
		vector<lower=-8, upper=8>[n_nodes] log_a_nodes;
		real log_c1;
		real log_c2;
		real log_c3;
		real x0;
}

transformed parameters {
		vector[n_pix] log_bg_int;
		vector[n_pix] log_gd1_int;
		vector[n_pix] xmod;

		vector[n_nodes] tmp;
		vector[2] xy;
		vector[2] node_xy;

		vector[n_nodes] phi2_nodes;

		real ln_bg_val;

		// Un-log some things
		vector[n_nodes] w_nodes;
		vector[n_nodes] a_nodes;
		real c1;
		real c2;
		real c3;

    matrix[2,2] C[n_nodes];

		w_nodes = exp(log_w_nodes);
		// a_nodes = exp(log_a_nodes);
		c1 = exp(log_c1);
		c2 = exp(log_c2);
		c3 = exp(log_c3);

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

            // Get the local covariance matrix and rotate to tangent space
            C[j] = get_cov(h_nodes[j], w_nodes[j]);
            C[j] = R[j] * C[j] * R[j]'; // R C R_T
						tmp[j] = log_a_nodes[j] + multi_normal_lpdf(xy | node_xy, C[j]);
				}

				log_bg_int[i] = ln_bg_quadratic_uniform(x[i], y[i],
                                                a1, b1, a2, b2,
                                                c1, c2, c3, x0);

		    log_gd1_int[i] = log_sum_exp(tmp);
				xmod[i] = log_sum_exp(log_gd1_int[i], log_bg_int[i]) + log_S[i];
	  }
}
model {
    // Priors
    for (n in 1:n_nodes) {
        log_w_nodes[n] ~ normal(log(0.15), 0.35)T[-2, -0.5];
        d_phi2_nodes[n] ~ normal(0, 0.2)T[-0.75, 0.75];
    }
    // target += -log_diff_exp(normal_lcdf(-0.5| log(0.15), 0.35),
    //                         normal_lcdf(-2| log(0.15), 0.35));
    target += -log_a_nodes; // like a regularization term to force amplitudes to 0

    //Likelihood
    hh ~ poisson_log(xmod);
}
