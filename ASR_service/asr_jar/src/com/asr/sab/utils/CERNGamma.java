package com.asr.sab.utils;

import org.apache.commons.math3.special.Gamma;

/*
 *  Copyright 2009 Heinrich Schuchardt.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *  under the License.
 *  
 *  
 * Copyright 1999 CERN - European Organization for Nuclear Research.
 * Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
 * is hereby granted without fee, provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear in supporting documentation.
 * CERN makes no representations about the suitability of this software for any purpose.
 * It is provided "as is" without expressed or implied warranty.
 */
public class CERNGamma {

	final static double MAXLOG = 709.782712893384;
	final static double MACHEP = 1.1102230246251565E-16;
	public static final double big = 4.503599627370496E15;
	public static final double biginv = 2.220446049250313E-16;

	public static double gammaIncInv(double P, double a) throws Exception {
		double x, x1, x2;
		double dP, phi;
		double n;
		double step, step1;
		double nmax = 10000; // maximum number of iterations

		if (P >= 1.0) {
			throw new Exception("propability too high");
		} else if (P < 0.0) {
			throw new Exception("negative propability");
		} else if (P == 0.0) {
			return 0.0;
		}

		x1 = Math.exp((Gamma.logGamma(a) + Math.log(P)) / a);
		x2 = -Math.log(1. - P) + Gamma.logGamma(a);
		x = .5 * (x1 + x2);

		if (P < 0.05) {
			x = x1;
		}
		if (P > 0.95) {
			x = x2;
		}

		dP = 0.;
		for (n = 0; n < nmax; n++) {
			dP = P - incompleteGamma(x, a);

			if (Math.abs(dP) <= MACHEP * P) {
				break;
			}

			phi = gamma_pdf(x, a, 1);

			step = dP / Math.max(2 * Math.abs(dP / x), phi);
			step1 = -((a - 1) / x - 1) * step * step / 4.0;

			if (Math.abs(step1) < Math.abs(step)) {
				step += step1;
			}

			if (x + step > 0) {
				x += step;
			} else {
				x /= 2.0;
			}
			// if (Math.abs(dP) < 1e-7) {
			// }
		}

		if (Math.abs(dP) > 1e-5) {
			throw new Exception("inverse failed to converge: a = " + a + ", p =" + P);
		}
		return x;
	}

	/**
	 * Propability density function for the Gamma distribution
	 *
	 * @param x
	 * @param a
	 * @param b
	 * @return
	 */
	public static double gamma_pdf(double x, double a, double b) {
		double r;
		if (x < 0) {
			return 0;
		}
		if (x == 0) {
			r = (a == 1) ? 1 / b : 0;
			return r;
		}
		if (a == 1) {
			r = Math.exp(-x / b) / b;
			return r;
		}
		r = Math.exp((a - 1) * Math.log(x / b) - x / b - Gamma.logGamma(a)) / b;
		return r;
	}

	/**
	 * Returns the Incomplete Gamma function
	 * 
	 * @param x
	 *            the integration end point.
	 * @param a
	 *            the parameter of the gamma distribution.
	 *
	 *            http://www.ssfnet.org/download/ssfnet_raceway-2.0.tar.gz
	 *
	 *            Copyright ï¿½ 1999 CERN - European Organization for Nuclear
	 *            Research. Permission to use, copy, modify, distribute and sell
	 *            this software and its documentation for any purpose is hereby
	 *            granted without fee, provided that the above copyright notice
	 *            appear in all copies and that both that copyright notice and this
	 *            permission notice appear in supporting documentation. CERN makes
	 *            no representations about the suitability of this software for any
	 *            purpose. It is provided "as is" without expressed or implied
	 *            warranty.
	 */
	public static double incompleteGamma(double x, double a) {

		double ans, ax, c, r;

		if (x <= 0 || a <= 0) {
			return 0.0;
		}

		if (x > 1.0 && x > a) {
			return 1.0 - incompleteGammaComplement(x, a);
		}

		/* Compute x**a * exp(-x) / gamma(a) */
		ax = a * Math.log(x) - x - Gamma.logGamma(a);
		if (ax < -MAXLOG) {
			return 0.0;
		}

		ax = Math.exp(ax);

		/* power series */
		r = a;
		c = 1.0;
		ans = 1.0;
		do {
			r += 1.0;
			c *= x / r;
			ans += c;
		} while (c / ans > MACHEP);

		return (ans * ax / a);

	}

	/**
	 * Returns the Complemented Incomplete Gamma function
	 * 
	 * @param a
	 *            the parameter of the gamma distribution.
	 * @param x
	 *            the integration start point.
	 *
	 *            http://www.ssfnet.org/download/ssfnet_raceway-2.0.tar.gz
	 *
	 *            Copyright ï¿½ 1999 CERN - European Organization for Nuclear
	 *            Research. Permission to use, copy, modify, distribute and sell
	 *            this software and its documentation for any purpose is hereby
	 *            granted without fee, provided that the above copyright notice
	 *            appear in all copies and that both that copyright notice and this
	 *            permission notice appear in supporting documentation. CERN makes
	 *            no representations about the suitability of this software for any
	 *            purpose. It is provided "as is" without expressed or implied
	 *            warranty.
	 */
	public static double incompleteGammaComplement(double x, double a) {
		double ans, ax, c, yc, r, t, y, z;
		double pk, pkm1, pkm2, qk, qkm1, qkm2;

		if (x <= 0 || a <= 0) {
			return 1.0;
		}

		if (x < 1.0 || x < a) {
			return 1.0 - incompleteGamma(x, a);
		}

		ax = a * Math.log(x) - x - Gamma.logGamma(a);
		if (ax < -MAXLOG) {
			return 0.0;
		}

		ax = Math.exp(ax);

		/* continued fraction */
		y = 1.0 - a;
		z = x + y + 1.0;
		c = 0.0;
		pkm2 = 1.0;
		qkm2 = x;
		pkm1 = x + 1.0;
		qkm1 = z * x;
		ans = pkm1 / qkm1;

		do {
			c += 1.0;
			y += 1.0;
			z += 2.0;
			yc = y * c;
			pk = pkm1 * z - pkm2 * yc;
			qk = qkm1 * z - qkm2 * yc;
			if (qk != 0) {
				r = pk / qk;
				t = Math.abs((ans - r) / r);
				ans = r;
			} else {
				t = 1.0;
			}

			pkm2 = pkm1;
			pkm1 = pk;
			qkm2 = qkm1;
			qkm1 = qk;
			if (Math.abs(pk) > big) {
				pkm2 *= biginv;
				pkm1 *= biginv;
				qkm2 *= biginv;
				qkm1 *= biginv;
			}
		} while (t > MACHEP);

		return ans * ax;
	}

}
