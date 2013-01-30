#ifndef LIBTENSOR_GEN_BTO_COMPARE_H
#define LIBTENSOR_GEN_BTO_COMPARE_H

#include <sstream>
#include <libtensor/defs.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/transf_list.h>
#include "gen_block_tensor_i.h"
#include "gen_block_tensor_ctrl.h"

namespace libtensor {


/** \brief Compares two general block tensors
    \tparam N Tensor order.
    \tparam Traits Traits class for this block tensor operation.

    This operation takes two block tensors with the same block %index space,
    compares them, and returns a structure that contains the first
    difference found.

    The constructor takes two block tensors, compare() performs the actual
    comparison and returns true if the block tensors are identical, false
    otherwise. When compare() returns false, the difference structure is
    available from get_diff().

    Along with two block tensors, the constructor takes the threshold for
    comparing data (the elements of the block tensors may not differ by
    more than the absolute value of the threshold) as well as a boolean
    parameter that enables or disables strict zero block comparison policy.

    When the strict zero block policy is on, the blocks marked as zero
    blocks in one of the block tensors must also be marked zero in the other
    one. When the policy is off, and a block is marked zero in one of the
    block tensors, it can be either marked zero in the other one or contain
    all zero elements within the threshold.

    The symmetries of the compared block tensors must yield identical
    orbits.

    Upon return from compare(), if differences are found, a structure is
    filled with data describing them.

    The main element of the difference structure is the kind of difference.
    Possible values are:
     - diff::DIFF_NODIFF - No differences found. Other elements of the
        structure have default values and are meaningless.
     - diff::DIFF_ORBLSTSZ - Orbit lists have different sizes. Further
        comparison is not performed, other members of the structure are
        meaningless.
     - diff::DIFF_ORBIT - Difference found in orbit lists. \c bidx
        identifies the block %index where the difference was found, and
        \c can1 and \c can2 contain whether the block is canonical.
     - diff::DIFF_DATA - Difference found in the canonical block data.
        \c bidx identifies the block, \c idx reads the position at which
        the difference is found. \c zero1 and \c zero2 specify whether
        the block is zero in the two tensors. \c data1 and \c data2 give
        the values that are different (only when \c zero1=false and
        \c zero2=false).

    Two special static methods tostr() will output the difference structure
    to a stream or a string in a human-readable format.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_tensor_type<N>::type -- Type of temporary
            block tensor
    - \c template to_compare_type<N>::type -- Type of tensor
            operation to_compare

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_compare : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

public:
    struct diff {
        enum {
            DIFF_NODIFF, //!< No differences found
            DIFF_ORBLSTSZ, //!< Different orbit list sizes
            DIFF_ORBIT, //!< Different orbits
            DIFF_TRANSF, //!< Different transformation within orbit
            DIFF_DATA //!< Difference in data
        };

        unsigned kind;
        index<N> bidx;
        index<N> idx;
        bool can1, can2;
        bool zero1, zero2;
        element_type data1, data2;
    };

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bt1; //!< First block %tensor
    gen_block_tensor_rd_i<N, bti_traits> &m_bt2; //!< Second block %tensor
    element_type m_thresh; //!< Threshold
    bool m_strict; //!< Strict zero blocks
    diff m_diff; //!< Difference structure

public:
    /** \brief Initializes the operation
        \param bt1 First %tensor.
        \param bt2 Second %tensor.
        \param thresh Equality threshold.
        \param strict Strict check of zero blocks.

        The two block tensors must have compatible block index spaces,
        otherwise an exception will be thrown.
     **/
    gen_bto_compare(
        gen_block_tensor_rd_i<N, bti_traits> &bt1,
        gen_block_tensor_rd_i<N, bti_traits> &bt2,
        const element_type &thresh = Traits::zero(),
        bool strict = true);

    /** \brief Performs the comparison
        \return \c true if all the elements are equal within
            the threshold, \c false otherwise
     **/
    bool compare();

    /** \brief Returns the difference structure
     **/
    const diff &get_diff() const {

        return m_diff;
    }

    /** \brief Prints the contents of the difference structure to
            a stream in a human-readable form
     **/
    void tostr(std::ostream &s);

    /** \brief Appends the contents of the difference structure in
            a human-readable form to the end of the string
     **/
    void tostr(std::string &s);

private:
    /** \brief Checks that two orbits have the same canonical %index
     **/
    bool compare_canonical(const abs_index<N> &acidx1,
        orbit<N, element_type> &o1, orbit<N, element_type> &o2);

    /** \brief Checks that the same transformation corresponds to a
            given %index
     **/
    bool compare_transf(const abs_index<N> &aidx,
        orbit<N, element_type> &o1, transf_list<N, element_type> &trl1,
        orbit<N, element_type> &o2, transf_list<N, element_type> &trl2);

    /** \brief Compares two canonical blocks identified by an %index
     **/
    bool compare_data(const abs_index<N> &aidx,
        gen_block_tensor_rd_ctrl<N, bti_traits> &ctrl1,
        gen_block_tensor_rd_ctrl<N, bti_traits> &ctrl2);
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COMPARE_H
