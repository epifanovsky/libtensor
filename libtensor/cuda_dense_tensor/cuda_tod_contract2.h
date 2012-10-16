#ifndef LIBTENSOR_CUDA_TOD_CONTRACT2_H
#define LIBTENSOR_CUDA_TOD_CONTRACT2_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>
#include <libtensor/tod/contraction2.h>
#include <libtensor/kernels/loop_list_node.h>


namespace libtensor {


/** \brief General contraction two CUDA dense tensors
    \tparam N Order of first tensor (A) less contraction degree.
    \tparam M Order of second tensor (B) less contraction degree.
    \tparam K Contraction degree (number of inner indexes).

    For more details see tod_contract2.

    \sa dense_tensor_i, contraction2, tod_contract2

    \ingroup libtensor_cuda_dense_tensor_tod
 **/
template<size_t N, size_t M, size_t K>
class cuda_tod_contract2 :
    public timings< cuda_tod_contract2<N, M, K> >,
    public noncopyable {

public:
    static const char *k_clazz;

public:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M //!< Order of result (C)
    };

private:
    struct args {
        contraction2<N, M, K> contr; //!< Contraction
        dense_tensor_rd_i<k_ordera, double> &ta; //!< First tensor (A)
        dense_tensor_rd_i<k_orderb, double> &tb; //!< Second tensor (B)
        double d; //!< Scaling factor

        args(
            const contraction2<N, M, K> &contr_,
            dense_tensor_rd_i<k_ordera, double> &ta_,
            dense_tensor_rd_i<k_orderb, double> &tb_,
            double d_) :
            contr(contr_), ta(ta_), tb(tb_), d(d_) { }
    };

    struct aligned_args : public args {
        permutation<k_ordera> perma;
        permutation<k_orderb> permb;
        permutation<k_orderc> permc;

        aligned_args(
            const args &ar_) :
            args(ar_) { }
        aligned_args(
            const args &ar_,
            const permutation<k_ordera> &perma_,
            const permutation<k_orderb> &permb_,
            const permutation<k_orderc> &permc_) :
            args(ar_), perma(perma_), permb(permb_), permc(permc_) { }
    };

    class loop_list_adapter {
    private:
        typedef std::list< loop_list_node<2, 1> > list_t;
        list_t &m_list;

    public:
        loop_list_adapter(list_t &list) : m_list(list) { }
        void append(size_t weight, size_t inca, size_t incb,
            size_t incc) {
            typedef typename list_t::iterator iterator_t;
            typedef loop_list_node<2, 1> node_t;
            iterator_t inode = m_list.insert(m_list.end(), node_t(weight));
            inode->stepa(0) = inca;
            inode->stepa(1) = incb;
            inode->stepb(0) = incc;
        }
    };

private:
    to_contract2_dims<N, M, K> m_dimsc; //!< Dimensions of result
    std::list<args> m_argslst; //!< List of arguments

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param ka Scalar transformation of A.
        \param tb Second contracted tensor B.
        \param kb Scalar transformation of B.
        \param kc Scalar transformation of result (default 1.0).
     **/
    cuda_tod_contract2(
        const contraction2<N, M, K> &contr,
        dense_tensor_rd_i<k_ordera, double> &ta,
        const scalar_transf<double> &ka,
        dense_tensor_rd_i<k_orderb, double> &tb,
        const scalar_transf<double> &kb,
        const scalar_transf<double> &kc = scalar_transf<double>());

    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param tb Second contracted tensor B.
        \param d Scaling factor d (default 1.0).
     **/
    cuda_tod_contract2(
        const contraction2<N, M, K> &contr,
        dense_tensor_rd_i<k_ordera, double> &ta,
        dense_tensor_rd_i<k_orderb, double> &tb,
        double d = 1.0);

    /** \brief Adds a set of arguments to the argument list
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param ka Scalar transformation of A.
        \param tb Second contracted tensor B.
        \param kb Scalar transformation of B.
        \param kc Scalar transformation of result (C).
     **/
    void add_args(
        const contraction2<N, M, K> &contr,
        dense_tensor_rd_i<k_ordera, double> &ta,
        const scalar_transf<double> &ka,
        dense_tensor_rd_i<k_orderb, double> &tb,
        const scalar_transf<double> &kb,
        const scalar_transf<double> &kc);

    /** \brief Adds a set of arguments to the argument list
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param tb Second contracted tensor B.
        \param d Scaling factor d.
     **/
    void add_args(
        const contraction2<N, M, K> &contr,
        dense_tensor_rd_i<k_ordera, double> &ta,
        dense_tensor_rd_i<k_orderb, double> &tb,
        double d);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Performs the operation
        \param zero Zero output before computing.
        \param d Scaling factor.
        \param tc Output tensor.
     **/
    void perform(bool zero, dense_tensor_wr_i<k_orderc, double> &tc);

private:
    void align(const sequence<2 * (N + M + K), size_t> &conn,
        permutation<N + K> &perma, permutation<M + K> &permb,
        permutation<N + M> &permc);

    void perform_internal(aligned_args &ar, double *pc,
        const dimensions<k_orderc> &dimsc);
};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_CONTRACT2_H

