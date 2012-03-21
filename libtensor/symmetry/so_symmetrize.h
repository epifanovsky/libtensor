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

	This symmetry operation symmetrizes a set of indexes that is specified by
	two sequences. The first sequence specifies the index groups that are
	going to be symmetrized, the second sequence specifies which index in the
	first index group will be symmetrized w.r.t to which indexes from the
	other index groups. E.g. the sequences [0112323] and [0122112]
    would result in symmetrization of three index groups as follows:
        [ijklmno] <-> [ijkonml] <-> [inlojmk]
            <-> [inlkmjo] <-> [imoknjl] <-> [imoljnk]

    Indexes for which the sequences are zero are not symmetrized (here: i).
    Beside zero each number in either sequence has to occur equally often in
    the sequence. The highest number in the first sequences is how often each
    number (beside zero) has to occur in the second sequence, and vice versa.

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
    bool m_symm; //!< Symmetric/anti-symmetric flag (for pair permutations)

public:
    /**	\brief Initializes the operation
		\param sym1 Symmetry container.
		\param idxgrp Index groups
		\param symidx Symmtrization indexes
		\param symm Symmetric (true)/anti-symmetric (false)
     **/
    so_symmetrize(const symmetry<N, T> &sym1,
            const sequence<N, size_t> &idxgrp,
            const sequence<N, size_t> &symidx, bool symm);

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
    bool symm; //!< Symmetrize/anti-symmetrize flag (for pair permutation)
    symmetry_element_set<N, T> &grp2; //!< Symmetry group 2 (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const sequence<N, size_t> &idxgrp_,
            const sequence<N, size_t> &symidx_, bool symm_,
            symmetry_element_set<N, T> &grp2_) :

                grp1(grp1_), idxgrp(idxgrp_), symidx(symidx_),
                symm(symm_), grp2(grp2_) { }

    virtual ~symmetry_operation_params() { }
};

} // namespace libtensor

#include "so_symmetrize_handlers.h"

#endif // LIBTENSOR_SO_SYMMETRIZE_H
