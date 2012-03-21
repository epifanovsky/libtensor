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

template<size_t N, size_t M, size_t K, typename T>
class so_merge;

template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_params< so_merge<N, M, K, T> >;

/**	\brief Merges multiple dimensions of a %symmetry group into one
	\tparam N Order of the argument space.
	\tparam M Dimensions to merge.
	\tparam K Number of separate merges.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space
	of order N - M + K.

    M dimensions of the original %symmetry group are merged in K steps to
    yield K dimensions in the result %symmetry group. The dimensions merged
    in each step are specified by K masks which have to be disjoint (i.e. no
    two masks can have the same dimension set to true).

    Since merging dimensions only makes sense if each mask comprises at least
    two dimensions, M should always be large or equal to 2 * K (although this
    is not explicitly checked).

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class so_merge : public symmetry_operation_base< so_merge<N, M, K, T> > {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef so_merge<N, M, K, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Input symmetry.
    mask<N> m_msk[K]; //!< K masks.
    size_t m_msk_set; //!< Number of masks set

public:
    /** \brief Constructor
		\param sym1 Input symmetry.
     **/
    so_merge(const symmetry<N, T> &sym1) :
        m_sym1(sym1), m_msk_set(0) { }

    /** \brief Add the next mask.

        This function adds the next mask to the operation. It has to be called
        exactly K times.
     **/
    void add_mask(const mask<N> &msk);

    void perform(symmetry<N - M + K, T> &sym2);
};

template<size_t N, size_t M, typename T>
class so_merge<N, M, M, T> :
    public symmetry_operation_base< so_merge<N, M, M, T> > {
public:
    static const char *k_clazz; //!< Class name

private:
    typedef so_merge<N, M, M, T> operation_t;
    typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
    const symmetry<N, T> &m_sym1; //!< Input symmetry.

public:
    /** \brief Constructor
        \param sym1 Input symmetry.
     **/
    so_merge(const symmetry<N, T> &sym1) :
        m_sym1(sym1) { }

    /** \brief Add the next mask.

        This function adds the next mask to the operation. It has to be called
        exactly K times.
     **/
    void add_mask(const mask<N> &msk) { }

    void perform(symmetry<N, T> &sym2) {
        so_copy<N, T>(m_sym1).perform(sym2);
    }
};

template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_params< so_merge<N, M, K, T> > :
public symmetry_operation_params_i {
public:
    const symmetry_element_set<N, T> &grp1; //!< Symmetry group (input)
    mask<N> msk[K]; //!< Masks
    symmetry_element_set<N - M + K, T> &grp2; //!< Symmetry group (output)

public:
    symmetry_operation_params(
            const symmetry_element_set<N, T> &grp1_,
            const mask<N> (&msk_)[K],
            symmetry_element_set<N - M + K, T> &grp2_) :

                grp1(grp1_), grp2(grp2_) {

        for (register size_t k = 0; k < K; k++) {
            msk[k] = msk_[k];
        }
    }

    virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_merge_handlers.h"

#endif // LIBTENSOR_SO_MERGE_H

