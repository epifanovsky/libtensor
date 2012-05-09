#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include "scalar_transf_double.h"
#include <libtensor/block_tensor/bto/bto_copy.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


struct btod_copy_traits : public bto_traits<double> {
};


template<size_t N>
class btod_copy : public bto_copy<N, btod_copy_traits> {
private:
    typedef bto_copy<N, btod_copy_traits> bto_copy_t;
    typedef typename bto_copy_t::scalar_tr_t scalar_tr_t;

public:
    btod_copy(block_tensor_i<N, double> &bta, double c = 1.0) :
        bto_copy_t(bta, scalar_tr_t(c)) {
    }

    btod_copy(block_tensor_i<N, double> &bta,
            const permutation<N> &p, double c = 1.0) :
        bto_copy_t(bta, p, scalar_tr_t(c)) {
    }

    virtual ~btod_copy() { }

private:
    btod_copy(const btod_copy<N>&);
    btod_copy<N> &operator=(const btod_copy<N>&);
};


} // namespace libtensor


#endif // LIBTENSOR_BTOD_COPY_H
t_schedule<N, double> m_sch;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the copy operation
		\param bt Source block %tensor.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bta, double c = 1.0);

	/**	\brief Initializes the permuted copy operation
		\param bt Source block %tensor.
		\param p Permutation.
		\param c Scaling coefficient.
	 **/
	btod_copy(block_tensor_i<N, double> &bta, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_copy() { }
	//@}

	//!	\name Implementation of
	//!		libtensor::direct_block_tensor_operation<N, double>
	//@{
	virtual const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<N, double> &get_symmetry() const {
		return m_sym;
	}

	using additive_btod<N>::perform;

	virtual void sync_on();
	virtual void sync_off();

	//@}

	//!	\name Implementation of libtensor::btod_additive<N>
	//@{
	virtual const assignment_schedule<N, double> &get_schedule() const {
		return m_sch;
	}
	//@}

protected:
	virtual void compute_block(bool zero, tensor_i<N, double> &blk,
		const index<N> &ib, const transf<N, double> &tr, double c,
		cpu_pool &cpus);

private:
	static block_index_space<N> mk_bis(const block_index_space<N> &bis,
		const permutation<N> &perm);
	void make_schedule();

private:
	btod_copy(const btod_copy<N>&);
	btod_copy<N> &operator=(const btod_copy<N>&);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_copy_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_BTOD_COPY_H
