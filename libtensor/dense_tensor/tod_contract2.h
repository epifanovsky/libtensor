#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/tod/contraction2.h>
#include <libtensor/kernels/loop_list_node.h>
#include "to_contract2_dims.h"


namespace libtensor {


/** \brief General contraction two dense tensors
    \tparam N Order of first tensor (A) less contraction degree.
    \tparam M Order of second tensor (B) less contraction degree.
    \tparam K Contraction degree (number of inner indexes).

    This operation performs the contraction of two tensors. The result is
    scaled by the given factor and added to the output tensor.

    The contraction is specified by passing an initialized contraction2 object.

    Contractions can be done in the streaming mode, which allows for multiple
    contractions to be accumulated into one result. Such contraction is
    initialized by passing the first set of arguments upon construction and
    adding further sets by calling add_args(). When running multiple
    contractions at once, the algorithm makes more efficient use of internal
    buffers, which leads to higher performance.

    \sa dense_tensor_i, contraction2

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, size_t M, size_t K>
class tod_contract2 : public timings< tod_contract2<N, M, K> > {
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
        dense_tensor_i<k_ordera, double> &ta; //!< First tensor (A)
        dense_tensor_i<k_orderb, double> &tb; //!< Second tensor (B)
        double d; //!< Scaling factor

        args(
            const contraction2<N, M, K> &contr_,
            dense_tensor_i<k_ordera, double> &ta_,
            dense_tensor_i<k_orderb, double> &tb_,
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
        \param tb Second contracted tensor B.
        \param d Scaling factor d (default 1.0).
     **/
    tod_contract2(
        const contraction2<N, M, K> &contr,
        dense_tensor_i<k_ordera, double> &ta,
        dense_tensor_i<k_orderb, double> &tb,
        double d = 1.0);

    /** \brief Adds a set of arguments to the argument list
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param tb Second contracted tensor B.
        \param d Scaling factor d.
     **/
    void add_args(
        const contraction2<N, M, K> &contr,
        dense_tensor_i<k_ordera, double> &ta,
        dense_tensor_i<k_orderb, double> &tb,
        double d);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Performs the operation
        \param zero Zero output before computing.
        \param d Scaling factor.
        \param tc Output tensor.
     **/
    void perform(bool zero, double d, dense_tensor_i<k_orderc, double> &tc);

private:
    void align(const sequence<2 * (N + M + K), size_t> &conn,
        permutation<N + K> &perma, permutation<M + K> &permb,
        permutation<N + M> &permc);

    void perform_internal(aligned_args &ar, double d, double *pc,
        const dimensions<k_orderc> &dimsc);

private:
    tod_contract2(const tod_contract2&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

