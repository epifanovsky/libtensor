#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H

#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    const tensor_transf<N, element_type> &m_tra;
    const symmetry<N, element_type> &m_symb;
    const dimensions<N> &m_bidimsb;
    index<N> m_ia;
    gen_block_stream_i<N, bti_traits> &m_out;

public:
    gen_bto_contract2_task(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        const symmetry<N, element_type> &symb,
        const dimensions<N> &bidimsb,
        const index<N> &ia,
        gen_block_stream_i<N, bti_traits> &out);

    virtual ~gen_bto_contract2_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2<N, M, K, Traits, Timed> &m_bto;
    const symmetry<N, element_type> &m_symb;
    gen_block_stream_i<N, bti_traits> &m_out;
    dimensions<N> m_bidimsb;
    gen_block_tensor_rd_ctrl<N, bti_traits> m_ca;
    orbit_list<N, element_type> m_ola;
    typename orbit_list<N, element_type>::iterator m_ioa;

public:
    gen_bto_contract2_task_iterator(
        gen_bto_contract2<N, M, K, Traits, Timed> &bto,
        const symmetry<N, element_type> &symb,
        gen_block_stream_i<N, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

private:
    void skip_zero_blocks();

};


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_contract2<N, M, K, Traits, Timed>::perform(
    const std::vector<size_t> &blst,
    gen_block_stream_i<NC, bti_traits> &out) {

    gen_bto_contract2::start_timer();

    try {

        out.open();

        gen_bto_copy_task_iterator<N, Traits> ti(m_bta, m_tra, m_symb, out);
        gen_bto_copy_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        gen_bto_contract2::stop_timer();
        throw;
    }

    gen_bto_contract2::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_IMPL_H
