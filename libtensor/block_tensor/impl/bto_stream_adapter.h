#ifndef LIBTENSOR_BTO_STREAM_ADAPTER_H
#define LIBTENSOR_BTO_STREAM_ADAPTER_H

namespace libtensor {


template<size_t N, typename Traits>
class bto_stream_adapter : public gen_block_stream_i<N, typename Traits::bti_traits> {
public:
    typedef typename Traits::bti_traits bti_traits;
    typedef typename bti_traits::element_type element_type;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

private:
    bto_stream_i<N, Traits> &m_out;

public:
    bto_stream_adapter(bto_stream_i<N, Traits> &out) : m_out(out) { }
    virtual ~bto_stream_adapter() { }
    virtual void open() { m_out.open(); }
    virtual void close() { m_out.close(); }
    virtual void put(
        const index<N> &idx,
        rd_block_type &blk,
        const tensor_transf<N, element_type> &tr) {
        m_out.put(idx, blk, tr);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_STREAM_ADAPTER_H
