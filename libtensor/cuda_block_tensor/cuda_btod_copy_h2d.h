#ifndef LIBTENSOR_CUDA_BTOD_COPY_H
#define LIBTENSOR_CUDA_BTOD_COPY_H

#include "cuda_block_tensor_traits.h"
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_h2d.h>

namespace libtensor {


/** \brief Copies a block tensor from host to device memory
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class cuda_btod_copy_h2d {
private:
	block_tensor_rd_i<N, double> &m_host_tensor; //!< Source %tensor

public:
    static const char *k_clazz; //!< Class name
//
public:
    typedef typename cuda_block_tensor_traits::bti_traits bti_traits;
//
//private:
//    gen_bto_copy< N, btod_traits, btod_copy<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Source block tensor (A) in host memory.
     **/
    cuda_btod_copy_h2d(block_tensor_rd_i<N, double> &bth) :

    	m_host_tensor(bth)  {

    }

    ~cuda_btod_copy_h2d() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

//    virtual const block_index_space<N> &get_bis() const {
//
//        return m_gbto.get_bis();
//    }
//
//    virtual const symmetry<N, double> &get_symmetry() const {
//
//        return m_gbto.get_symmetry();
//    }
//
//    virtual const assignment_schedule<N, double> &get_schedule() const {
//
//        return m_gbto.get_schedule();
//    }

    //@}



    void perform(cuda_block_tensor_wr_i<N, bti_traits> &btd) {

    	 start_timer("perform");

    	    try {

    	        gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_host_tensor);
    	        dimensions<N> bidimsa = m_host_tensor.get_bis().get_block_index_dims();


    	        std::vector<size_t> nzorba;
    	        ca.req_nonzero_blocks(nzorba);

    	        gen_block_tensor_wr_ctrl<N, bti_traits> cb(btd);
				//dimensions<N> bidimsb = btd.get_bis().get_block_index_dims();


				std::vector<size_t> nzorbb;
				cb.req_nonzero_blocks(nzorbb);

    	        for(size_t i = 0; i < nzorba.size(); i++) {

    	        	cuda_tod_copy_h2d(nzorba[i]).perform(nzorbb[i]);

    	        }

    	    } catch(...) {
    	        stop_timer("perform");
    	        throw;
    	    }

    	    stop_timer("perform");

    }

    };


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_COPY_H
