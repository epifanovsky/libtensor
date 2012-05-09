#ifndef LIBTENSOR_SO_MERGE_H
#define LIBTENSOR_SO_MERGE_H


#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "bad_symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"
#include "so_copy.h"


namespace libtensor {


template<size_t N, size_t M, typename T>
class so_merge;


template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_merge<N, M, T> >;


/** \brief Merges multiple dimensions of a %symmetry group into one
    \tparam N Order of the argument space.
    \tparam M Decrement in dimensions.

    The operation takes a %symmetry group that is defined for a %tensor
    space of order N and produces a group that acts in a %tensor space
    of order N - M by doing a number of merging steps.

    The mask specifies the total number of dimensions that are merged, while
    the sequence specifies the dimensions per merging step: all dimensions for
    which the mask is true and the sequence has the same value are merged onto
    one result dimension. The merging steps in the sequence have to be numbered
    consecutively starting from zero. Thus, for k merging steps the highest
    value of the sequence should be k - 1, and the number of masked dimensions
    should be M + k.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_merge : public symmetry_operation_base< so_merge<N, M, T> > {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef so_merge<N, M, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Input symmetry.
    mask<N> m_msk; //!< Total mask.
    sequence<N, size_t> m_mseq; //!< Sequence of merging steps

public:
    /** \brief Constructor
        \param sym1 Input symmetry.
        \param msk Total mask.
        \param mseq Sequence of merging steps
     **/
    so_merge(const symmetry<N, T> &sym1, const mask<N> &msk,
            const sequence<N, size_t> &mseq);

    void perform(symmetry<N - M, T> &sym2);
};


/** \brief Specialization of so_merge for M = 0 (no merge)
 **/
template<size_t N, typename T>
class so_merge<N, 0, T> {
private:
    const symmetry<N, T> &m_sym1; //!< Input symmetry.

public:
    so_merge(const symmetry<N, T> &sym1, const mask<N> &msk,
            const sequence<N, size_t> &mseq) :
        m_sym1(sym1) { }

    void perform(symmetry<N, T> &sym2) {
        so_copy<N, T>(m_sym1).perform(sym2);
    }
};


template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_merge<N, M, T> > :
public symmetry_operation_params_i {
public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group (input)
    mask<N> msk; //!< Mask
    sequence<N, size_t> mseq; //!< Merging sequence
    symmetry_element_set<N - M, T> &grp2; //!< Symmetry group (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const mask<N> &msk_,
            const sequence<N, size_t> &mseq_,
            symmetry_element_set<N - M, T> &grp2_) :

                grp1(grp1_), msk(msk_), mseq(mseq_), grp2(grp2_) {
    }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor


#include "so_merge_handlers.h"


#endif // LIBTENSOR_SO_MERGE_H

