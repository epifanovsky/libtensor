#ifndef LIBTENSOR_BTOD_CONTRACT3_H
#define LIBTENSOR_BTOD_CONTRACT3_H

#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/btod_contract2.h>

namespace libtensor {


/** \brief Contracts a train of three tensors
    \tparam N1 Order of first tensor less first contraction degree.
    \tparam N2 Order of second tensor less total contraction degree.
    \tparam N3 Order of third tensor less second contraction degree.
    \tparam K1 First contraction degree.
    \tparam K2 Second contraction degree.

    This algorithm computes the contraction of three linearly connected tensors.

    The contraction is performed as follows. The first tensor is contracted
    with the second tensor to form an intermediate, which is then contracted
    with the third tensor to yield the final result.

    The formation of the intermediate is done in batches:
    \f[
        ABC = A(B_1 + B_2 + \dots + B_n)C = \sum_{i=1}^n (AB_i)C \qquad
        B = \sum_{i=1}^n B_i
    \f]

    \ingroup libtensor_block_tensor
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
class btod_contract3 : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    contraction2<N1, N2 + K2, K1> m_contr1; //!< First contraction
    contraction2<N1 + N2, N3, K2> m_contr2; //!< Second contraction
    block_tensor_i<N1 + K1, double> &m_bta; //!< First argument (A)
    block_tensor_i<N2 + K1 + K2, double> &m_btb; //!< Second argument (B)
    block_tensor_i<N3 + K2, double> &m_btc; //!< Third argument (C)

private:
    class batch_ab_task : public libutil::task_i {
    private:
        btod_contract2<N1, N2 + K2, K1> &m_contr;
        index<N1 + N2 + K2> m_idx;
        block_tensor_i<N1 + N2 + K2, double> &m_btab;

    public:
        batch_ab_task(
            btod_contract2<N1, N2 + K2, K1> &contr,
            const index<N1 + N2 + K2> &idx,
            block_tensor_i<N1 + N2 + K2, double> &btab) :
            m_contr(contr), m_idx(idx), m_btab(btab) { }
        virtual ~batch_ab_task() { }
        virtual void perform();
    };

    class batch_ab_task_iterator : public libutil::task_iterator_i {
    private:
        std::vector<batch_ab_task*> &m_tl;
        typename std::vector<batch_ab_task*>::iterator m_i;

    public:
        batch_ab_task_iterator(std::vector<batch_ab_task*> &tl) :
            m_tl(tl), m_i(m_tl.begin()) { }
        virtual ~batch_ab_task_iterator() { }
        virtual bool has_more() const;
        virtual libutil::task_i *get_next();
    };

    class batch_ab_task_observer : public libutil::task_observer_i {
    public:
        virtual void notify_start_task(libutil::task_i *t) { }
        virtual void notify_finish_task(libutil::task_i *t) { }
    };

public:
    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param btc Third tensor argument (C).
     **/
    btod_contract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        block_tensor_i<N1 + K1, double> &bta,
        block_tensor_i<N2 + K1 + K2, double> &btb,
        block_tensor_i<N3 + K2, double> &btc);

    /** \brief Computes the contraction
     **/
    void perform(block_tensor_i<N1 + N2 + N3, double> &btd);

private:
    void compute_batch_ab(
        btod_contract2<N1, N2 + K2, K1> &contr,
        const std::vector<size_t> &blst,
        block_tensor_i<N1 + N2 + K2, double> &btab);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT3_H

