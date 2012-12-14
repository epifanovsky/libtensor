#ifndef LIBTENSOR_CUDA_TOD_COPY_D2H_H
#define LIBTENSOR_CUDA_TOD_COPY_D2H_H

#include <libtensor/timings.h>
#include <cuda_runtime_api.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/bad_dimensions.h>

namespace libtensor {


/** \brief Copy tensor from host to device

    \ingroup libtensor_cuda_tod
 **/
template<size_t N>
class cuda_tod_copy_d2h :
public timings< cuda_tod_copy_d2h<N> > {
	dense_tensor_rd_i<N, double> &m_dev_tensor; //!< Source %tensor
private:

public:
	static const char *k_clazz; //!< Class name
	enum {
	        k_orderc = N //!< Order of tensors (C)
	    };

public:
    /** \brief Initializes the handle
     **/
    cuda_tod_copy_d2h(dense_tensor_rd_i<N, double> &dev_tensor);

    /** \brief Frees the handle
     **/
    ~cuda_tod_copy_d2h() {}

public:
    /** \brief Perform copying
     **/
    void perform(dense_tensor_wr_i<N, double> &host_tensor);

    /** \brief Perform actual copying
    **/
    void do_perform(dense_tensor_wr_i<N, double> &host_tensor);

};

template<size_t N>
const char *cuda_tod_copy_d2h<N>::k_clazz = "cuda_tod_copy_d2h<N>";

template<size_t N>
cuda_tod_copy_d2h<N>::cuda_tod_copy_d2h(dense_tensor_rd_i<N, double> &dev_tensor) :
	m_dev_tensor(dev_tensor) {

}



template<size_t N>
void cuda_tod_copy_d2h<N>::perform(dense_tensor_wr_i<N, double> &host_tensor)  {
	static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

		if(!host_tensor.get_dims().equals(m_dev_tensor.get_dims())) {
			throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
				"host_tensor");
		}
		do_perform(host_tensor);
}

template<size_t N>
void cuda_tod_copy_d2h<N>::do_perform(dense_tensor_wr_i<N, double> &host_tensor) {

	cuda_tod_copy_d2h<N>::start_timer();

	try {

	dense_tensor_rd_ctrl<k_orderc, double> cd(m_dev_tensor);
	dense_tensor_wr_ctrl<k_orderc, double> ch(host_tensor);
	ch.req_prefetch();
	cd.req_prefetch();

	const double *pd = cd.req_const_dataptr();
	double *ph = ch.req_dataptr();

	cuda_tod_copy_d2h<N>::start_timer("copy_d2h");
	cudaError_t ec = cudaMemcpy(ph, pd, sizeof(double) * m_dev_tensor.get_dims().get_size(), cudaMemcpyDeviceToHost);
	if(ec != cudaSuccess) {
	//	throw cuda_exception(k_clazz, method, __FILE__, __LINE__,
		//	cudaGetErrorString(ec));
	}
	cuda_tod_copy_d2h<N>::stop_timer("copy_d2h");

	cd.ret_const_dataptr(pd);
	ch.ret_dataptr(ph);

	} catch(...) {
		cuda_tod_copy_d2h<N>::stop_timer();
		throw;
	}
	cuda_tod_copy_d2h<N>::stop_timer();
}

} // namespace libtensor




#endif // LIBTENSOR_CUDA_TOD_COPY_D2H_H
