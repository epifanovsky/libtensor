#ifndef LIBTENSOR_SO_REDUCE_H
#define LIBTENSOR_SO_REDUCE_H

#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "bad_symmetry.h"
#include "so_copy.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_reduce;

template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_reduce<N, M, T> >;


/**	\brief Projection of a %symmetry group onto a subspace
	\tparam N Order of the argument space.
	\tparam M Decrement in the order of the result space.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space
	of order N - M by doing a number of reduction steps.

	The mask specifies the total number of dimensions which are reduced (thus
	it has to have M entries set to true), while the sequence specifies the
	dimensions per reduction steps: all dimensions for which the mask is true
	and the sequence has the same value are reduced together. The reduction
	steps in the sequence have to be numbered consecutively starting from
	zero. The %index range specifies the blocks in the tensor over which the
	reductions are performed, i.e. also here the dimensions for which the mask
	is false are ignored. The range for dimensions belonging to the same
	reduction step have to be identical.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_reduce : public symmetry_operation_base< so_reduce<N, M, T> > {

private:
    typedef so_reduce<N, M, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

public:
    static const char *k_clazz;

private:
    const symmetry<N, T> &m_sym1; //!< Input symmetry
    mask<N> m_msk; //!< Total reduction mask
    sequence<N, size_t> m_rseq; //!< Sequence of reduction steps
    index_range<N> m_rblrange; //!< Block index range of reduction steps
    index_range<N> m_riblrange; //!< In-block index range of reduction steps

public:
    so_reduce(const symmetry<N, T> &sym1,
            const mask<N> &msk, const sequence<N, size_t> &rseq,
            const index_range<N> &rblrange, const index_range<N> &riblrange);

    void perform(symmetry<N - M, T> &sym2);

};

/**	\brief Projection of a %symmetry group onto vacuum (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_reduce<N, N, T> {
public:
    so_reduce(const symmetry<N, T> &sym1,
            const mask<N> &msk, const sequence<N, size_t> &rseq,
            const index_range<N> &rblrange, const index_range<N> &riblrange)
    { }

    void perform(symmetry<0, T> &sym2) {
        sym2.clear();
    }
};

/**	\brief Projection of a %symmetry group onto itself (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_reduce<N, 0, T> {
private:
    const symmetry<N, T> &m_sym1;
public:
    so_reduce(const symmetry<N, T> &sym1,
            const mask<N> &msk, const sequence<N, size_t> &rseq,
            const index_range<N> &rblrange,  const index_range<N> &riblrange) :
        m_sym1(sym1) { }


    void perform(symmetry<N, T> &sym2) {
        so_copy<N, T>(m_sym1).perform(sym2);
    }
};

template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_reduce<N, M, T> > :
public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group
    mask<N> msk; //!< Mask
    sequence<N, size_t> rseq; //!< Reduction sequence
    index_range<N> rblrange; //!< Reduction block index range
    index_range<N> riblrange; //!< Reduction in-block index range
    symmetry_element_set<N - M, T> &grp2;

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const mask<N> &msk_, const sequence<N, size_t> &rseq_,
            const index_range<N> &rblrange_, const index_range<N> &riblrange_,
            symmetry_element_set<N - M, T> &grp2_) :
                grp1(grp1_), msk(msk_), rseq(rseq_),
                rblrange(rblrange_), riblrange(riblrange_), grp2(grp2_) {
    }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_reduce_handlers.h"

#endif // LIBTENSOR_SO_REDUCE_H

