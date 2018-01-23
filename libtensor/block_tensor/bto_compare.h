#ifndef LIBTENSOR_BTO_COMPARE_H
#define LIBTENSOR_BTO_COMPARE_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_compare.h>
#include <libtensor/block_tensor/bto_traits.h>

namespace libtensor {


/** \brief Compares two block tensors
    \tparam N Tensor order.

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

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_compare : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename gen_bto_compare<N, bto_traits<T> >::diff diff;

private:
    gen_bto_compare<N, bto_traits<T> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bt1 First %tensor.
        \param bt2 Second %tensor.
        \param thresh Equality threshold.
        \param strict Strict check of zero blocks.

        The two block tensors must have compatible block index spaces,
        otherwise an exception will be thrown.
     **/
    bto_compare(
            block_tensor_rd_i<N, T> &bt1,
            block_tensor_rd_i<N, T> &bt2,
            T thresh = 0.0, bool strict = true);

    /** \brief Performs the comparison
        \return \c true if all the elements are equal within
            the threshold, \c false otherwise
     **/
    bool compare();

    /** \brief Returns the difference structure
     **/
    const diff &get_diff() const {

        return m_gbto.get_diff();
    }

    /** \brief Prints the contents of the difference structure to
            a stream in a human-readable form
     **/
    void tostr(std::ostream &s) {

        m_gbto.tostr(s);
    }

    /** \brief Appends the contents of the difference structure in
            a human-readable form to the end of the string
     **/
    void tostr(std::string &s) {
        m_gbto.tostr(s);
    }
};

template<size_t N>
using btod_compare = bto_compare<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_BTO_COMPARE_H
