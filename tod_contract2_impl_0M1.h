#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_0M1_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_0M1_H

#include "defs.h"
#include "exception.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts a first-order %tensor with a (M+1)-th order %tensor
		over one index.

	Performs contractions:
	\f[ c_{ij \cdots m} = \mathcal{P}_c \sum_{p}
		\mathcal{P}_a a_{p} \mathcal{P}_b b_{ij \cdots mp} \f]
	\f[ c_{ij \cdots m} = c_{ij \cdots m} + d \cdot \mathcal{P}_c \sum_{p}
		\mathcal{P}_a a_{p} \mathcal{P}_b b_{ij \cdots mp} \f]
	\f[ \underbrace{ij \cdots m}_{M} \f]

	\ingroup libtensor_tod
**/
template<size_t M>
class tod_contract2_impl<0,M,1> {
public:
	/**	\brief \f$ c_{ij \cdots m} = \mathcal{P}_c \sum_{p}
			\mathcal{P}_a a_{p} \mathcal{P}_b b_{ij \cdots mp} \f$
	**/
	static void contract(double *c, const dimensions<M> &dc,
		const permutation<M> &pcc, const double *a,
		const dimensions<1> &da, const permutation<1> &pca,
		const double *b, const dimensions<M+1> &db,
		const permutation<M+1> &pcb) throw(exception);

	/**	\brief \f$ c_{ij \cdots m} = c_{ij \cdots m} +
			d \cdot \mathcal{P}_c \sum_{p}
			\mathcal{P}_a a_{p} \mathcal{P}_b b_{ij \cdots mp} \f$
	**/
	static void contract(double *c, const dimensions<M> &dc,
		const permutation<M> &pcc, const double *a,
		const dimensions<1> &da, const permutation<1> &pca,
		const double *b, const dimensions<M+1> &db,
		const permutation<M+1> &pcb, double d) throw(exception);

private:
	/**	\brief \f$ c_{ij \cdots m} = \sum_{p}
			a_{p} b_{ij \cdots mp} \f$
	**/
	static void c_012_0_0123(double *c, const dimensions<M> &dc,
		const double *a, const dimensions<1> &da,
		const double *b, const dimensions<M+1> &db) throw(exception);

	/**	\brief \f$ c_{ij \cdots m} = c_{ij \cdots m} +
			d \sum_{p} a_{p} b_{ij \cdots mp} \f$
	**/
	static void c_012_0_0123(double *c, const dimensions<M> &dc,
		const double *a, const dimensions<1> &da,
		const double *b, const dimensions<M+1> &db, double d)
			throw(exception);
};

template<size_t M>
void tod_contract2_impl<0,M,1>::contract(
	double *c, const dimensions<M> &dc, const permutation<M> &pc,
	const double *a, const dimensions<1> &da, const permutation<1> &pa,
	const double *b, const dimensions<M+1> &db, const permutation<M+1> &pb)
	throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_012_0_0123(c, dc, a, da, b, db);
	} else {
		throw_exc("tod_contract2_impl<0,M,1>", "contract()",
			"Contraction not implemented");
	}
}

template<size_t M>
void tod_contract2_impl<0,M,1>::contract(
	double *c, const dimensions<M> &dc, const permutation<M> &pc,
	const double *a, const dimensions<1> &da, const permutation<1> &pa,
	const double *b, const dimensions<M+1> &db, const permutation<M+1> &pb,
	double d) throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_012_0_0123(c, dc, a, da, b, db, d);
	} else {
		throw_exc("tod_contract2_impl<0,M,1>", "contract()",
			"Contraction not implemented");
	}
}

template<size_t M>
void tod_contract2_impl<0,M,1>::c_012_0_0123(double *c, const dimensions<M> &dc,
	const double *a, const dimensions<1> &da, const double *b,
	const dimensions<M+1> &db) throw(exception) {

	size_t sza = da.get_size(), szc = dc.get_size();
	const double *pb = b;
	double *pc = c;
	for(size_t i=0; i<szc; i++) {
		*pc = cblas_ddot(sza, a, 1, pb, 1);
		pc++; pb+=sza;
	}
}

template<size_t M>
void tod_contract2_impl<0,M,1>::c_012_0_0123(double *c, const dimensions<M> &dc,
	const double *a, const dimensions<1> &da, const double *b,
	const dimensions<M+1> &db, double d) throw(exception) {

	size_t sza = da.get_size(), szc = dc.get_size();
	const double *pb = b;
	double *pc = c;
	if(d == 1.0) {
		for(size_t i=0; i<szc; i++) {
			*pc += cblas_ddot(sza, a, 1, pb, 1);
			pc++; pb+=sza;
		}
	} else if(d == -1.0) {
		for(size_t i=0; i<szc; i++) {
			*pc -= cblas_ddot(sza, a, 1, pb, 1);
			pc++; pb+=sza;
		}
	} else {
		for(size_t i=0; i<szc; i++) {
			*pc += d*cblas_ddot(sza, a, 1, pb, 1);
			pc++; pb+=sza;
		}
	}
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_0M1_H

