#ifndef LIBTENSOR_BTOD_MULT_H
#define LIBTENSOR_BTOD_MULT_H

namespace libtensor {


/**	\brief Elementwise multiplication of two block tensors
	\tparam N Tensor order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult {
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_bta; //!< First argument
	block_tensor_i<N, double> &m_btb; //!< Second argument

public:
	btod_mult(block_tensor_i<N, double> &bta,
		block_tensor_i<N, double> &btb);

	void perform(block_tensor_i<N, double> &btc);

	void perform(block_tensor_i<N, double> &btc, double c);

private:
	btod_mult(const btod_mult<N> &);
	const btod_mult<N> &operator=(const btod_mult<N> &);

};


template<size_t N>
const char *btod_mult<N>::k_clazz = "btod_mult<N>";


template<size_t N>
btod_mult<N>::btod_mult(block_tensor_i<N, double> &bta,
	block_tensor_i<N, double> &btb) :

	m_bta(bta), m_btb(btb) {

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H