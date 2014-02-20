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

/** \brief Symmetrizes groups of indexes.
    \tparam N Symmetry cardinality (%tensor order).
    \tparam T Tensor element type.

    This symmetry operation symmetrizes a set of dimensions as specified by two
    sequences, \c idxgrp and \c symidx. The first sequence defines groups of
    dimensions such that the dimensions within in a group are not permuted by
    the symmetrization, but one index group is permuted with other index groups
    as whole. The second sequences defines the order of the dimensions within
    each group, i.e. it defines which dimension is first, second, third, ...
    in a group (and for each group). The dimensions are then symmetrized so
    that the dimensions which are marked first in the groups are permuted among
    each other, as well as those which are marked second, third, etc. Indexes
    for which the sequences are zero are not symmetrized. As consequence,
    all non-zero numbers have to occur equally often in any sequence.

    For illustration two examples:
    Example 1:
        Symmetrization of \c [ijkl]<->[klij] would require as sequences
        \c idxgrp=[1122] and \c symidx=[1212].

    Example 2:
        The symmetrization of a 7-index symmetry object given by the sequences
        \c idxgrp=[0112323] and \c symidx=[0122112]
        would result in:
        \c [ijklmno]<->[ijkonml]<->[inlojmk]
        \c <->[inlkmjo]<->[imoknjl]<->[imoljnk]

    In addition to the two sequences also two scalar transformations have
    to be specified. These are associated to pair permutations and the
    cyclic permutation, respectively, which form a generating set of the
    symmetry group for the symmetrization. In example 2, the pair
    and cyclic permutation would be:
        \c [0123456->0536142] and \c [0123456->0532416]

    (Example 1 does not have a cyclic permutation, but only a pair
    permutation!)

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
    /** \brief Initializes the operation
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

    /** \brief Performs the operation
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
    const symmetry_element_set<N, T> &g1; //!< Symmetry group 1
    sequence<N, size_t> idxgrp; //!< Sequence of index groups
    sequence<N, size_t> symidx; //!< Sequences of symmterizations
    scalar_transf<T> trp; //!< Transformation for pair perm
    scalar_transf<T> trc; //!< Transformation for cyclic perm
    symmetry_element_set<N, T> &g2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &g1_,
            const sequence<N, size_t> &idxgrp_,
            const sequence<N, size_t> &symidx_,
            const scalar_transf<T> &trp_, const scalar_transf<T> &trc_,
            symmetry_element_set<N, T> &g2_) :

                g1(g1_), idxgrp(idxgrp_), symidx(symidx_),
                trp(trp_), trc(trc_), g2(g2_) { }

    virtual ~symmetry_operation_params() { }
};

} // namespace libtensor

#include "so_symmetrize_handlers.h"

#endif // LIBTENSOR_SO_SYMMETRIZE_H
