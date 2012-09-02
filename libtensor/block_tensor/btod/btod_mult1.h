#ifndef LIBTENSOR_BTOD_MULT1_H
#define LIBTENSOR_BTOD_MULT1_H

#include <libtensor/timings.h>
#include <libtensor/core/block_tensor_i.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \sa tod_mult1

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult1 :
    public timings< btod_mult1<N> >{
public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_i<N, double> &m_btb; //!< Second argument
    permutation<N> m_pb; //!< Permutation of second argument
    bool m_recip; //!< Reciprocal
    double m_c; //!< Scaling coefficient

public:
    btod_mult1(block_tensor_i<N, double> &btb,
            bool recip = false, double c = 1.0);

    btod_mult1(block_tensor_i<N, double> &btb, const permutation<N> &pb,
            bool recip = false, double c = 1.0);

    void perform(block_tensor_i<N, double> &btc);

    void perform(block_tensor_i<N, double> &btc, double c);

protected:
    void compute_block(dense_tensor_i<N, double> &blk, const index<N> &idx);


private:
    void do_perform(block_tensor_i<N, double> &btc, bool zero, double c);

private:
    btod_mult1(const btod_mult1<N> &);
    const btod_mult1<N> &operator=(const btod_mult1<N> &);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_H
