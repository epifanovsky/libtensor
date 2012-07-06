#ifndef LIBTENSOR_TOD_COPY_H
#define LIBTENSOR_TOD_COPY_H

#include <libtensor/timings.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/tod/loop_list_add.h>
#include <libtensor/tod/loop_list_copy.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Copies the contents of a tensor, permutes and scales the entries if
        necessary
    \tparam N Tensor order.

    This operation makes a transformed copy of a %tensor.
    The result can replace or be added to the output %tensor.

    <b>Examples</b>

    Plain copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    tod_copy<2> cp(t1);
    cp.perform(t2); // Copies the elements of t1 to t2
    \endcode

    Scaled copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    tod_copy<2> cp(t1, 0.5);
    cp.perform(t2); // Copies the elements of t1 multiplied by 0.5 to t2
    \endcode

    Permuted copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    tod_copy<2> cp(t1, perm);
    cp.perform(t2); // Copies transposed t1 to t2
    \endcode

    Permuted and scaled copy:
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    tod_copy<2> cp(t1, perm, 0.5);
    cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
    \endcode
    or
    \code
    dense_tensor_i<2, double> &t1(...), &t2(...);
    permutation<2> perm; perm.permute(0, 1); // Sets up a permutation
    tensor_transf<2, double> tr(perm, 0.5);
    tod_copy<2> cp(t1, tr);
    cp.perform(t2); // Copies transposed t1 scaled by 0.5 to t2
    \endcode

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_copy :
    public loop_list_add,
    public loop_list_copy,
    public timings< tod_copy<N> > {

public:
    static const char *k_clazz; //!< Class name

    typedef tensor_transf<N, double> tensor_transf_t;

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< Source tensor
    tensor_transf_t m_tr; //!< Permutation of indexes
    dimensions<N> m_dimsb; //!< Dimensions of output tensor

public:
    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param tr Tensor transformation.
     **/
    tod_copy(dense_tensor_rd_i<N, double> &ta,
            const tensor_transf_t &tr = tensor_transf_t());

    /** \brief Prepares the copy operation
        \param ta Source tensor.
        \param c Coefficient.
     **/
    tod_copy(dense_tensor_rd_i<N, double> &ta, double c);

    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param p Permutation of tensor indexes.
        \param c Coefficient.
     **/
    tod_copy(dense_tensor_rd_i<N, double> &ta, const permutation<N> &p,
        double c = 1.0);


    /** \brief Virtual destructor
     **/
    virtual ~tod_copy() { }

    /** \brief Prefetches the source tensor
     **/
    void prefetch();

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param c Scaling coefficient.
        \param tb Output tensor.
     **/
    void perform(bool zero, double c, dense_tensor_wr_i<N, double> &tb);

    //@}

private:
    /** \brief Creates the dimensions of the output using an input
            tensor and a permutation of indexes
     **/
    static dimensions<N> mk_dimsb(dense_tensor_rd_i<N, double> &ta,
        const permutation<N> &perm);

    template<typename Base>
    void do_perform(double c, dense_tensor_wr_i<N,double> &t);

    template<typename Base>
    void build_loop(typename Base::list_t &loop, const dimensions<N> &dimsa,
        const permutation<N> &perma, const dimensions<N> &dimsb);

private:
    /** \brief Private copy constructor
     **/
    tod_copy(const tod_copy&);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_H
