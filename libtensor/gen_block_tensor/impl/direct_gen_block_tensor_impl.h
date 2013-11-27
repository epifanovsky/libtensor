#ifndef LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_IMPL_H
#define LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_IMPL_H

#include <libutil/threads/auto_lock.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include "block_map_impl.h"


namespace libtensor {


template<size_t N, typename BtiTraits>
class direct_gen_block_tensor_task : public libutil::task_i {
public:
    typedef typename BtiTraits::template wr_block_type<N>::type wr_block_type;

private:
    direct_gen_bto<N, BtiTraits> &m_op;
    index<N> m_idx;
    wr_block_type &m_blk;

public:
    direct_gen_block_tensor_task(direct_gen_bto<N, BtiTraits> &op,
            const index<N> &idx, wr_block_type &blk) :
        m_op(op), m_idx(idx), m_blk(blk) { }

    virtual ~direct_gen_block_tensor_task() { }

    virtual unsigned long get_cost() const { return 0; }

    void perform() {
        m_op.compute_block(m_idx, m_blk);
    }
};


template<size_t N, typename BtiTraits>
class direct_gen_block_tensor_task_iterator : public libutil::task_iterator_i {
private:
    direct_gen_block_tensor_task<N, BtiTraits> &m_t;
    bool m_done;

public:
    direct_gen_block_tensor_task_iterator(
            direct_gen_block_tensor_task<N, BtiTraits> &t) :
        m_t(t), m_done(false) { }

    virtual ~direct_gen_block_tensor_task_iterator() { }

    virtual bool has_more() const {
        return !m_done;
    }

    virtual libutil::task_i *get_next() {
        m_done = true;
        return &m_t;
    }
};


class direct_gen_block_tensor_task_observer : public libutil::task_observer_i {
public:
    virtual ~direct_gen_block_tensor_task_observer() { }
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) { }
};


template<size_t N, typename BtTraits>
const char *direct_gen_block_tensor<N, BtTraits>::k_clazz =
    "direct_gen_block_tensor<N, BtTraits>";


template<size_t N, typename BtTraits>
direct_gen_block_tensor<N, BtTraits>::direct_gen_block_tensor(operation_t &op) :

    base_t(op), m_bidims(get_bis().get_block_index_dims()), m_map(get_bis()) {

}


template<size_t N, typename BtTraits>
bool direct_gen_block_tensor<N, BtTraits>::on_req_is_zero_block(
    const index<N> &idx) {

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    return !get_op().get_schedule().contains(idx);
}


template<size_t N, typename BtTraits>
void direct_gen_block_tensor<N, BtTraits>::on_req_nonzero_blocks(
    std::vector<size_t> &nzlst) {

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    nzlst.clear();
    const assignment_schedule<N, element_type> &sch = get_op().get_schedule();
    for(typename assignment_schedule<N, element_type>::iterator i = sch.begin();
        i != sch.end(); ++i) {
        nzlst.push_back(sch.get_abs_index(i));
    }
}


template<size_t N, typename BtTraits>
typename direct_gen_block_tensor<N, BtTraits>::rd_block_type &
direct_gen_block_tensor<N, BtTraits>::on_req_const_block(
    const index<N> &idx) {

    typedef typename BtTraits::template block_type<N>::type block_type;

    static const char *method = "on_req_const_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

#ifdef LIBTENSOR_DEBUG
    if(!get_op().get_schedule().contains(idx)) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "idx");
    }
#endif // LIBTENSOR_DEBUG

    abs_index<N> aidx(idx, m_bidims);
    typename std::map<size_t, size_t>::iterator icnt =
        m_count.insert(std::make_pair(aidx.get_abs_index(), size_t(0))).first;
    bool newblock = icnt->second++ == 0;
    bool inprogress = m_inprogress.count(aidx.get_abs_index()) > 0;

    if(newblock) {
        m_map.create(idx);
    }

    block_type &blk = m_map.get(idx);

    if(newblock) {

        std::set<size_t>::iterator i =
                m_inprogress.insert(aidx.get_abs_index()).first;
        m_lock.unlock();
        try {
            get_op().compute_block(idx, blk);
        } catch(...) {
            m_lock.lock();
            throw;
        }
        m_lock.lock();
        m_inprogress.erase(i);
        m_cond.signal(aidx.get_abs_index());

    } else if(inprogress) {

        libutil::loaded_cond<size_t> cond(0);
        m_cond.insert(aidx.get_abs_index(), &cond);
        m_lock.unlock();
        try {
            libutil::thread_pool::release_cpu();
            cond.wait();
            libutil::thread_pool::acquire_cpu();
        } catch(...) {
            m_lock.lock();
            throw;
        }
        m_lock.lock();
        m_cond.erase(aidx.get_abs_index(), &cond);
    }

    return blk;
}


template<size_t N, typename BtTraits>
void direct_gen_block_tensor<N, BtTraits>::on_ret_const_block(
        const index<N> &idx) {

    static const char *method = "on_ret_const_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    abs_index<N> aidx(idx, m_bidims);
    typename std::map<size_t, size_t>::iterator icnt =
        m_count.find(aidx.get_abs_index());
    if(icnt == m_count.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "idx");
    }

    if(--icnt->second == 0) {
        m_map.remove(idx);
        m_count.erase(icnt);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_IMPL_H
