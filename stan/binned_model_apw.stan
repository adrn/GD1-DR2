functions{

	  matrix get_cov(real h, real w) {
			  matrix[2, 2] C;
				C[1, 1] = h*h;
				C[2, 1] = 0.;
				C[1, 2] = 0.;
				C[2, 2] = w*w;
				return C;
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

parameters {
		vector[n_nodes] d_phi2_nodes;
		vector<lower=-3, upper=0.5>[n_nodes] log_w_nodes;
		vector[n_nodes] log_a_nodes;
		real log_c1;
		real log_c2;
		real log_c3;
		real x0;
}

transformed parameters {
		// vector[n_pix] logbg_pix;
		vector[n_pix] logint_pix;
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
						tmp[j] = log_a_nodes[j] + multi_normal_lpdf(xy | node_xy, get_cov(h_nodes[j], w_nodes[j]));
				}

				ln_bg_val = ln_bg_quadratic_uniform(x[i], y[i],
																					  a1, b1, a2, b2,
																		 				c1, c2, c3, x0);

		    xmod[i] = log_sum_exp(tmp);
				xmod[i] = log_sum_exp(xmod[i], ln_bg_val);
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
