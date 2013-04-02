#ifndef LIBTENSOR_CUDA_TOD_COPY_H2D_H
#define LIBTENSOR_CUDA_TOD_COPY_H2D_H

#include <libtensor/timings.h>
#include <cuda_runtime_api.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/cuda_dense_tensor/cuda_dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/cuda_dense_tensor/cuda_dense_tensor_ctrl.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/cuda/cuda_utils.h>

namespace libtensor {


/** \brief Copy tensor from host to device

    \ingroup libtensor_cuda_tod
 **/
template<size_t N>
class cuda_tod_copy_h2d :
public timings< cuda_tod_copy_h2d<N> > {
	dense_tensor_rd_i<N, double> &m_host_tensor; //!< Source %tensor
private:

public:
	static const char *k_clazz; //!< Class name
	enum {
	        k_orderc = N //!< Order of tensors (C)
	    };

public:
    /** \brief Initializes the handle
     **/
    cuda_tod_copy_h2d(dense_tensor_rd_i<N, double> &host_tensor);

    /** \brief Frees the handle
     **/
    ~cuda_tod_copy_h2d() {}

public:
    /** \brief Perform copying
    **/
    void perform(cuda_dense_tensor_wr_i<N, double> &dev_tensor);

    /** \brief Perform actual copying
    **/
    void do_perform(cuda_dense_tensor_wr_i<N, double> &dev_tensor);

};

template<size_t N>
const char *cuda_tod_copy_h2d<N>::k_clazz = "cuda_tod_copy_h2d<N>";

template<size_t N>
cuda_tod_copy_h2d<N>::cuda_tod_copy_h2d(dense_tensor_rd_i<N, double> &host_tensor) :
	m_host_tensor(host_tensor) {

}



template<size_t N>
void cuda_tod_copy_h2d<N>::perform(cuda_dense_tensor_wr_i<N, double> &dev_tensor)  {
	static const char *method = "perform(dense_tensor_wr_i<N, double>&)";

		if(!dev_tensor.get_dims().equals(m_host_tensor.get_dims())) {
			throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
				"dev_tensor");
		}
		do_perform(dev_tensor);
}

template<size_t N>
void cuda_tod_copy_h2d<N>::do_perform(cuda_dense_tensor_wr_i<N, double> &dev_tensor) {


    static const char *method =
        "do_perform( dense_tensor_wr_i<N , double>&)";

	cuda_tod_copy_h2d<N>::start_timer();

	try {

	dense_tensor_rd_ctrl<N, double> ch(m_host_tensor);
	cuda_dense_tensor_wr_ctrl<N, double> cd(dev_tensor);
	ch.req_prefetch();
	cd.req_prefetch();

	const double *ph = ch.req_const_dataptr();
	cuda_pointer<double> pd = cd.req_dataptr();

	cuda_tod_copy_h2d<N>::start_timer("copy_h2d");
	cuda_utils::handle_error(
			cudaMemcpy(pd.get_physical_pointer(), ph, sizeof(double) * m_host_tensor.get_dims().get_size(), cudaMemcpyHostToDevice),
			g_ns, k_clazz, method, __FILE__, __LINE__);
	cuda_tod_copy_h2d<N>::stop_timer("copy_h2d");

	ch.ret_const_dataptr(ph);
	cd.ret_dataptr(pd);

	} catch(...) {
		cuda_tod_copy_h2d<N>::stop_timer();
		throw;
	}
	cuda_tod_copy_h2d<N>::stop_timer();
}

} // namespace libtensor




#endif // LIBTENSOR_CUDA_TOD_COPY_H2D_H
