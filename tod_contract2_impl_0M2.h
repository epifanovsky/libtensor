#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_0M2_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_0M2_H

#include "defs.h"
#include "exception.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts a second-order %tensor with a (M+2)-th order %tensor
		over two indexes.

	Performs contractions:
	\f[ c_{ij \cdots m} = \mathcal{P}_c \sum_{pq}
		\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ij \cdots mpq} \f]
	\f[ c_{ij \cdots m} = c_{ij \cdots m} + d \cdot \mathcal{P}_c \sum_{pq}
		\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ij \cdots mpq} \f]
	\f[ \underbrace{ij \cdots m}_{M} \f]

	\ingroup libtensor_tod
**/
template<size_t M>
class tod_contract2_impl<0,M,2> {
public:
	/**	\brief \f$ c_{ij \cdots m} = \mathcal{P}_c \sum_{pq}
			\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ij \cdots mpq} \f$
	**/
	static void contract(double *c, const dimensions<M> &dc,
		const permutation<M> &pcc, const double *a,
		const dimensions<2> &da, const permutation<2> &pca,
		const double *b, const dimensions<M+2> &db,
		const permutation<M+2> &pcb) throw(exception);

	/**	\brief \f$ c_{ij \cdots m} = c_{ij \cdots m} +
			d \cdot \mathcal{P}_c \sum_{pq}
			\mathcal{P}_a a_{pq} \mathcal{P}_b b_{ij \cdots mpq} \f$
	**/
	static void contract(double *c, const dimensions<M> &dc,
		const permutation<M> &pcc, const double *a,
		const dimensions<2> &da, const permutation<2> &pca,
		const double *b, const dimensions<M+2> &db,
		const permutation<M+2> &pcb, double d) throw(exception);

private:
	/**	\brief \f$ c_{ij \cdots m} = \sum_{pq}
			a_{pq} b_{ij \cdots mpq} \f$
	**/
	static void c_01_01_0123(double *c, const dimensions<M> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<M+2> &db) throw(exception);

	/**	\brief \f$ c_{ij \cdots m} = c_{ij \cdots m} +
			d \sum_{pq} a_{pq} b_{ij \cdots mpq} \f$
	**/
	static void c_01_01_0123(double *c, const dimensions<M> &dc,
		const double *a, const dimensions<2> &da,
		const double *b, const dimensions<M+2> &db, double d)
			throw(exception);
};

template<size_t M>
void tod_contract2_impl<0,M,2>::contract(
	double *c, const dimensions<M> &dc, const permutation<M> &pc,
	const double *a, const dimensions<2> &da, const permutation<2> &pa,
	const double *b, const dimensions<M+2> &db, const permutation<M+2> &pb)
	throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_01_0123(c, dc, a, da, b, db);
	} else {
		throw_exc("tod_contract2_impl<0,M,2>", "contract()",
			"Contraction not implemented");
	}
}

template<size_t M>
void tod_contract2_impl<0,M,2>::contract(
	double *c, const dimensions<M> &dc, const permutation<M> &pc,
	const double *a, const dimensions<2> &da, const permutation<2> &pa,
	const double *b, const dimensions<M+2> &db, const permutation<M+2> &pb,
	double d) throw(exception) {

	if(pc.is_identity() && pa.is_identity() && pb.is_identity()) {
		c_01_01_0123(c, dc, a, da, b, db, d);
	} else {
		throw_exc("tod_contract2_impl<0,M,2>", "contract()",
			"Contraction not implemented");
	}
}

template<size_t M>
void tod_contract2_impl<0,M,2>::c_01_01_0123(double *c, const dimensions<M> &dc,
	const double *a, const dimensions<2> &da, const double *b,
	const dimensions<M+2> &db) throw(exception) {

	size_t sza = da.get_size(), szc = dc.get_size();
	const double *pb = b;
	double *pc = c;
	for(size_t i=0; i<szc; i++) {
		*pc = cblas_ddot(sza, a, 1, pb, 1);
		pc++; pb+=sza;
	}
}

template<size_t M>
void tod_contract2_impl<0,M,2>::c_01_01_0123(double *c, const dimensions<M> &dc,
	const double *a, const dimensions<2> &da, const double *b,
	const dimensions<M+2> &db, double d) throw(exception) {

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

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_0M2_H

