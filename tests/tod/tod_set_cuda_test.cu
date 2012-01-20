#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/tod/tod_set_cuda.h>
#include "tod_set_cuda_test.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include "../compare_ref.h"

namespace libtensor {

const double tod_set_cuda_test::k_thresh = 1e-14;

typedef dense_tensor<4, double, std_allocator<double> > h_tensor4_d;
typedef dense_tensor<4, double, libvmm::cuda_allocator<double> > d_tensor4_d;

void tod_set_cuda_test::perform() throw(libtest::test_exception) {

	 std::ostringstream tnss;
	 tnss << "tod_set_cuda_test::perform()";
	 std::string tn = tnss.str();

    cpu_pool cpus(1);
    double set_num = 5.0;

    try {
	index<4> i1, i2;
	i2[0]=3; i2[1]=3; i2[2]=4; i2[3]=4;
	index_range<4> ir(i1, i2);
	dimensions<4> dim(ir);
	d_tensor4_d d_t(dim);
	h_tensor4_d h_t(dim), h_t_ref(dim);

	double ta_max = 0.0;
	{
		dense_tensor_ctrl<4, double> ctrlt_ref(h_t_ref);

		double *ptrt_ref = ctrlt_ref.req_dataptr();
		for(size_t i = 0; i < dim.get_size(); i++) {
			ptrt_ref[i] = set_num;
		}
		ta_max = std::max(ta_max, set_num);
		ctrlt_ref.ret_dataptr(ptrt_ref); ptrt_ref = 0;
	}

    tod_set_cuda<4> op(set_num);
	op.perform(cpus, d_t);

	 copyTensorDeviceToHost(d_t, h_t);
	compare_ref<4>::compare(tn.c_str(), h_t, h_t_ref, ta_max * k_thresh);

    } catch(exception &e) {
           fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }

}

template<typename T, size_t N>
void tod_set_cuda_test::copyTensorHostToDevice(dense_tensor<N, T, std_allocator<T> > &ht, dense_tensor<N, T, libvmm::cuda_allocator<T> > &dt)
{
	dense_tensor_ctrl<N, T> dtc(dt);
	dense_tensor_ctrl<N, T> htc(ht);
	T *hdta = htc.req_dataptr();
	T *ddta = dtc.req_dataptr();
	libvmm::cuda_allocator<T>::copy_to_device(ddta, hdta, ht.get_dims().get_size());

	htc.ret_dataptr(hdta); hdta = NULL;
	dtc.ret_dataptr(ddta); ddta = NULL;
}

template<typename T, size_t N>
void tod_set_cuda_test::copyTensorDeviceToHost(dense_tensor<N, T, libvmm::cuda_allocator<T> > &dt, dense_tensor<N, T, std_allocator<T> > &ht)
{
	dense_tensor_ctrl<N, T> htc(ht), dtc(dt);
	T *hdta = htc.req_dataptr();
	T *ddta = dtc.req_dataptr();
	libvmm::cuda_allocator<T>::copy_to_host(hdta, ddta, ht.get_dims().get_size());

	htc.ret_dataptr(hdta); hdta = NULL;
	dtc.ret_dataptr(ddta); ddta = NULL;
}

} // namespace libtensor

