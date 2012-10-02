#ifndef LIBTENSOR_TOD_APPLY_H
#define LIBTENSOR_TOD_APPLY_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/tod/loop_list_apply.h>

namespace libtensor {


/** \brief Applies a functor to all tensor elements and scales / permutes them
        before, if necessary
    \tparam N Tensor order.

    This operation applies the given functor to each tensor element,
    transforming the tensor before and after applying the functor.
    The result can replace or be added to the output tensor.

    A class to be used as functor needs to have
    1. a proper copy constructor
    \code
        Functor(const Functor &f);
    \endcode
    2. an implementation of the function
    \code
        double Functor::operator()(const double &a);
    \endcode

    The latter function should perform the intended operation of the functor
    on the tensor data.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename Functor>
class tod_apply :
    public loop_list_apply<Functor>,
    public timings< tod_apply<N, Functor> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef scalar_transf<double> scalar_transf_type;
    typedef tensor_transf<N, double> tensor_transf_type;

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< Source %tensor
    Functor m_fn; //!< Functor
    double m_c1; //!< Scaling coefficient before fn
    double m_c2; //!< Scaling coefficient after fn
    permutation<N> m_permb; //!< Permutation of result
    dimensions<N> m_dimsb; //!< Dimensions of output %tensor

public:
    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param tr1 Tensor transformation before
        \param tr2 Tensor transformation after
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn,
            const scalar_transf_type &tr1 = scalar_transf_type(),
            const tensor_transf_type &tr2 = tensor_transf_type());

    /** \brief Prepares the copy operation
        \param ta Source tensor.
        \param c Coefficient (apply before).
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn, double c);

    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param p Permutation of tensor elements (apply before).
        \param c Coefficient (apply before).
     **/
    tod_apply(dense_tensor_rd_i<N, double> &ta, const Functor &fn,
        const permutation<N> &p, double c = 1.0);

    /** \brief Performs the operation
        \param zero Zero result first
        \param tb Add result to
     **/
    void perform(bool zero, dense_tensor_wr_i<N, double> &t);

private:
    /** \brief Creates the dimensions of the output using an input
            tensor and a permutation of indexes
     **/
    static dimensions<N> mk_dimsb(dense_tensor_rd_i<N, double> &ta,
            const permutation<N> &perm);

    void build_loop(typename loop_list_apply<Functor>::list_t &loop,
            const dimensions<N> &dimsa, const permutation<N> &perma,
            const dimensions<N> &dimsb);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_APPLY_H
