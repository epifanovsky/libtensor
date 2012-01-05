#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/tod/tod_trace.h>
#include "../compare_ref.h"
#include "tod_trace_test.h"

namespace libtensor {


void tod_trace_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1(1);
	test_1(3);
	test_1(16);
	test_2(1);
	test_2(3);
	test_2(16);

	test_3(1, 1);
	test_3(1, 3);
	test_3(3, 1);
	test_3(3, 5);
	test_3(16, 16);
	test_4(1, 1);
	test_4(1, 3);
	test_4(3, 1);
	test_4(3, 5);
	test_4(16, 16);

	test_5(1, 1, 1);
	test_5(1, 1, 3);
	test_5(1, 3, 1);
	test_5(3, 1, 1);
	test_5(3, 7, 5);
	test_5(8, 8, 8);
	test_6(1, 1, 1);
	test_6(1, 1, 3);
	test_6(1, 3, 1);
	test_6(3, 1, 1);
	test_6(3, 7, 5);
	test_6(8, 8, 8);
}


/**	\test Computes the trace of a square matrix: \f$ d = \sum_i a_{ii} \f$
 **/
void tod_trace_test::test_1(size_t ni) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_trace_test::test_1(" << ni << ")";
	std::string tns = tnss.str();

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = ni - 1; i2[1] = ni - 1;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sza = dims.get_size();

	dense_tensor<2, double, allocator_t> ta(dims);

	double d_ref = 0.0;
	{
		dense_tensor_ctrl<2, double> tca(ta);

		double *pa = tca.req_dataptr();

		for(size_t i = 0; i < sza; i++) pa[i] = drand48();

		for(size_t i = 0; i < ni; i++) {
			index<2> ia; ia[0] = i; ia[1] = i;
			abs_index<2> aia(ia, dims);
			d_ref += pa[aia.get_abs_index()];
		}

		tca.ret_dataptr(pa); pa = 0;
	}

	double d = tod_trace<1>(ta).calculate();

	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(tns.c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a square matrix (with permutation):
		\f$ d = \sum_i a_{ii} \f$
 **/
void tod_trace_test::test_2(size_t ni) throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_trace_test::test_2(" << ni << ")";
	std::string tns = tnss.str();

	typedef std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = ni - 1; i2[1] = ni - 1;
	dimensions<2> dims(index_range<2>(i1, i2));
	size_t sza = dims.get_size();

	dense_tensor<2, double, allocator_t> ta(dims);

	double d_ref = 0.0;
	{
		dense_tensor_ctrl<2, double> tca(ta);

		double *pa = tca.req_dataptr();

		for(size_t i = 0; i < sza; i++) pa[i] = drand48();

		for(size_t i = 0; i < ni; i++) {
			index<2> ia; ia[0] = i; ia[1] = i;
			abs_index<2> aia(ia, dims);
			d_ref += pa[aia.get_abs_index()];
		}

		tca.ret_dataptr(pa); pa = 0;
	}

	permutation<2> perm; perm.permute(0, 1);
	double d = tod_trace<1>(ta, perm).calculate();

	if(fabs(d - d_ref) > fabs(d_ref * 1e-15)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(tns.c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor:
		\f$ d = \sum_{ij} a_{ijij} \f$
 **/
void tod_trace_test::test_3(size_t ni, size_t nj)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_trace_test::test_3(" << ni << ", " << nj << ")";
	std::string tns = tnss.str();

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = ni - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	size_t sza = dims.get_size();

	dense_tensor<4, double, allocator_t> ta(dims);

	double d_ref = 0.0;
	{
		dense_tensor_ctrl<4, double> tca(ta);

		double *pa = tca.req_dataptr();

		for(size_t i = 0; i < sza; i++) pa[i] = drand48();

		for(size_t i = 0; i < ni; i++) {
		for(size_t j = 0; j < nj; j++) {
			index<4> ia;
			ia[0] = i; ia[1] = j; ia[2] = i; ia[3] = j;
			abs_index<4> aia(ia, dims);
			d_ref += pa[aia.get_abs_index()];
		}
		}

		tca.ret_dataptr(pa); pa = 0;
	}

	double d = tod_trace<2>(ta).calculate();

	if(fabs(d - d_ref) > fabs(d_ref * 1e-14)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(tns.c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 4-index tensor
		(with permutation): \f$ d = \sum_{ij} a_{iijj} \f$
 **/
void tod_trace_test::test_4(size_t ni, size_t nj)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_trace_test::test_4(" << ni << ", " << nj << ")";
	std::string tns = tnss.str();

	typedef std_allocator<double> allocator_t;

	try {

	index<4> i1, i2;
	i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = nj - 1; i2[3] = nj - 1;
	dimensions<4> dims(index_range<4>(i1, i2));
	size_t sza = dims.get_size();

	dense_tensor<4, double, allocator_t> ta(dims);

	double d_ref = 0.0;
	{
		dense_tensor_ctrl<4, double> tca(ta);

		double *pa = tca.req_dataptr();

		for(size_t i = 0; i < sza; i++) pa[i] = drand48();

		for(size_t i = 0; i < ni; i++) {
		for(size_t j = 0; j < nj; j++) {
			index<4> ia;
			ia[0] = i; ia[1] = i; ia[2] = j; ia[3] = j;
			abs_index<4> aia(ia, dims);
			d_ref += pa[aia.get_abs_index()];
		}
		}

		tca.ret_dataptr(pa); pa = 0;
	}

	permutation<4> perm; perm.permute(1, 2);
	double d = tod_trace<2>(ta, perm).calculate();

	if(fabs(d - d_ref) > fabs(d_ref * 1e-14)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(tns.c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 6-index tensor:
		\f$ d = \sum_{ijk} a_{ijkijk} \f$
 **/
void tod_trace_test::test_5(size_t ni, size_t nj, size_t nk)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_trace_test::test_5(" << ni << ", " << nj << ", "
		<< nk << ")";
	std::string tns = tnss.str();

	typedef std_allocator<double> allocator_t;

	try {

	index<6> i1, i2;
	i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = ni - 1;
	i2[4] = nj - 1; i2[5] = nk - 1;
	dimensions<6> dims(index_range<6>(i1, i2));
	size_t sza = dims.get_size();

	dense_tensor<6, double, allocator_t> ta(dims);

	double d_ref = 0.0;
	{
		dense_tensor_ctrl<6, double> tca(ta);

		double *pa = tca.req_dataptr();

		for(size_t i = 0; i < sza; i++) pa[i] = drand48();

		for(size_t i = 0; i < ni; i++) {
		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++) {
			index<6> ia;
			ia[0] = i; ia[1] = j; ia[2] = k;
			ia[3] = i; ia[4] = j; ia[5] = k;
			abs_index<6> aia(ia, dims);
			d_ref += pa[aia.get_abs_index()];
		}
		}
		}

		tca.ret_dataptr(pa); pa = 0;
	}

	double d = tod_trace<3>(ta).calculate();

	if(fabs(d - d_ref) > fabs(d_ref * 1e-14)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(tns.c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


/**	\test Computes the trace of a matricized 6-index tensor
		(with permutation): \f$ d = \sum_{ijk} a_{kkjjii} \f$
 **/
void tod_trace_test::test_6(size_t ni, size_t nj, size_t nk)
	throw(libtest::test_exception) {

	std::ostringstream tnss;
	tnss << "tod_trace_test::test_6(" << ni << ", " << nj << ", "
		<< nk << ")";
	std::string tns = tnss.str();

	typedef std_allocator<double> allocator_t;

	try {

	index<6> i1, i2;
	i2[0] = nk - 1; i2[1] = nk - 1; i2[2] = nj - 1; i2[3] = nj - 1;
	i2[4] = ni - 1; i2[5] = ni - 1;
	dimensions<6> dims(index_range<6>(i1, i2));
	size_t sza = dims.get_size();

	dense_tensor<6, double, allocator_t> ta(dims);

	double d_ref = 0.0;
	{
		dense_tensor_ctrl<6, double> tca(ta);

		double *pa = tca.req_dataptr();

		for(size_t i = 0; i < sza; i++) pa[i] = drand48();

		for(size_t i = 0; i < ni; i++) {
		for(size_t j = 0; j < nj; j++) {
		for(size_t k = 0; k < nk; k++) {
			index<6> ia;
			ia[0] = k; ia[1] = k; ia[2] = j;
			ia[3] = j; ia[4] = i; ia[5] = i;
			abs_index<6> aia(ia, dims);
			d_ref += pa[aia.get_abs_index()];
		}
		}
		}

		tca.ret_dataptr(pa); pa = 0;
	}

	permutation<6> perm;
	perm.permute(0, 5).permute(1, 2); // kkjjii -> ijkjik
	perm.permute(3, 4); // ijkjik -> ijkijk
	double d = tod_trace<3>(ta, perm).calculate();

	if(fabs(d - d_ref) > fabs(d_ref * 1e-14)) {
		std::ostringstream ss;
		ss << "Result doesn't match reference: " << d << " (result), "
			<< d_ref << " (reference), " << d - d_ref << " (diff)";
		fail_test(tns.c_str(), __FILE__, __LINE__, ss.str().c_str());
	}

	} catch(exception &e) {
		fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
