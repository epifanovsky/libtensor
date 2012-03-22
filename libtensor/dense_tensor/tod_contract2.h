#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/mp/cpu_pool.h>
#include <libtensor/tod/contraction2.h>
#include <libtensor/kernels/loop_list_node.h>


namespace libtensor {


/** \brief Contracts two tensors (double)

    \tparam N Order of the first %tensor (a) less the contraction degree
    \tparam M Order of the second %tensor (b) less the contraction degree
    \tparam K Contraction degree (the number of indexes over which the
        tensors are contracted)

    This operation contracts %tensor T1 permuted as P1 with %tensor T2
    permuted as P2 over n last indexes. The result is permuted as Pres
    and written or added to the resulting %tensor.

    Although it is convenient to define a contraction through permutations,
    it is not the most efficient way of calculating it. This class seeks
    to use algorithms tailored for different tensors to get the best
    performance. For more information, read the wiki section on %tensor
    contractions.

    \ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_contract2 : public timings< tod_contract2<N, M, K> > {

public:
    static const char *k_clazz;

private:
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

public:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M //!< Order of result (C)
    };

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    dense_tensor_i<k_ordera, double> &m_ta; //!< First tensor (a)
    dense_tensor_i<k_orderb, double> &m_tb; //!< Second tensor (b)

public:
    /** \brief Initializes the contraction operation

        \param contr Contraction.
        \param ta Tensor a (first argument).
        \param tb Tensor b (second argument).
     **/
    tod_contract2(const contraction2<N, M, K> &contr,
        dense_tensor_i<k_ordera, double> &ta,
        dense_tensor_i<k_orderb, double> &tb);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Performs the operation
        \param cpus Pool of CPUs.
        \param zero Zero output before computing.
        \param d Scaling factor.
        \param tc Output tensor.
     **/
    void perform(cpu_pool &cpus, bool zero, double d,
        dense_tensor_i<k_orderc, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

