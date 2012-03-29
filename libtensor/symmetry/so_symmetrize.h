#ifndef LIBTENSOR_SO_SYMMETRIZE_H
#define LIBTENSOR_SO_SYMMETRIZE_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {

template<size_t N, typename T>
class so_symmetrize;

template<size_t N, typename T>
class symmetry_operation_params< so_symmetrize<N, T> >;

/**	\brief Symmetrizes groups of indexes.
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	This symmetry operation symmetrizes a number of indexes given by two
	sequences. The first sequence specifies the groups by which the indexes
	are symmetrized, while the second sequence yields the indexes in each
	group that are symmetrized w.r.t to each other. Indexes for which the
	sequences are zero are not symmetrized. Thus, each non-zero number in
	the sequences has to occur equally often in a sequences. E.g. given a
	7-index symmetry object the sequences [0112323] and [0122112] would
	result in symmetrization of three groups of indexes as follows:
        [ijklmno] <-> [ijkonml] <-> [inlojmk]
            <-> [inlkmjo] <-> [imoknjl] <-> [imoljnk]
    In addition to the two sequences also two scalar transformations have
    to be specified. These are associated to the pair permutations and the
    cyclic permutation, respectively, which form a generating set of the
    symmetry group for the symmetrization. In the above example, the pair
    and cyclic permutation are:
        [0123456->0536142] and [0123456->0532416]

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_symmetrize : public symmetry_operation_base< so_symmetrize<N, T> > {
private:
    static const char *k_clazz;

    typedef so_symmetrize<N, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Symmetry group
    sequence<N, size_t> m_idxgrp; //!< Index groups to be symmetrized
    sequence<N, size_t> m_symidx; //!< Symmetrization indexes
    const scalar_transf<T> &m_trp; //!< Transf for pair perm
    const scalar_transf<T> &m_trc; //!< Transf for cyclic perm

public:
    /**	\brief Initializes the operation
		\param sym1 Symmetry container.
		\param idxgrp Index groups
		\param symidx Symmtrization indexes
		\param pt Scalar transformation for pair permutation
		\param ct Scalar transformation for cyclic permutation
     **/
    so_symmetrize(const symmetry<N, T> &sym1,
            const sequence<N, size_t> &idxgrp,
            const sequence<N, size_t> &symidx,
            const scalar_transf<T> &pt,
            const scalar_transf<T> &ct);

    /**	\brief Performs the operation
		\param sym2 Destination %symmetry container.
     **/
    void perform(symmetry<N, T> &sym2);

private:
    so_symmetrize(const so_symmetrize<N, T>&);
    const so_symmetrize<N, T> &operator=(const so_symmetrize<N, T>&);
};

template<size_t N, typename T>
class symmetry_operation_params< so_symmetrize<N, T> > :
    public symmetry_operation_params_i {

public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group 1
    sequence<N, size_t> idxgrp; //!< Sequence of index groups
    sequence<N, size_t> symidx; //!< Sequences of symmterizations
    scalar_transf<T> trp; //!< Transformation for pair perm
    scalar_transf<T> trc; //!< Transformation for cyclic perm
    symmetry_element_set<N, T> &grp2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const sequence<N, size_t> &idxgrp_,
            const sequence<N, size_t> &symidx_,
            const scalar_transf<T> &trp_, const scalar_transf<T> &trc_,
            symmetry_element_set<N, T> &grp2_) :

                grp1(grp1_), idxgrp(idxgrp_), symidx(symidx_),
                trp(trp_), trc(trc_), grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};

} // namespace libtensor

#include "so_symmetrize_handlers.h"

#endif // LIBTENSOR_SO_SYMMETRIZE_H
