#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libvmm.h>
#include <libtensor.h>
#include "compare_ref.h"
#include "btod_random_test.h"

namespace libtensor {

void btod_random_test::perform() throw(libtest::test_exception) 
{

	typedef libvmm::std_allocator<double> allocator_t;
	typedef tensor<4, double, allocator_t> tensor_t;
	typedef tensor_ctrl<4, double> tensor_ctrl_t;
	typedef block_tensor<4, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<4, double> block_tensor_ctrl_t;

	try {

	index<4> i1, i2;
	i2[0] = 3; i2[1] = 4;	i2[2] = 3; i2[3] = 4;
	dimensions<4> dims(index_range<4>(i1, i2));
	block_index_space<4> bis(dims);
	mask<4> msk; 
	msk[0]=true; msk[1]=false; msk[2]=true; msk[3]=false; 
	bis.split(msk,2);
	msk[0]=false; msk[1]=true; msk[2]=false; msk[3]=true;
	bis.split(msk,2);
	dimensions<4> bidims = bis.get_block_index_dims();	

	block_tensor_t bta(bis);
	block_tensor_ctrl_t btactrl(bta);

	symel_cycleperm<4, double> cycle1(msk, bidims);
	msk[0]=true; msk[1]=false; msk[2]=true; msk[3]=false; 
	symel_cycleperm<4, double> cycle2(msk, bidims);
	
	btactrl.req_sym_add_element(cycle1);
	btactrl.req_sym_add_element(cycle2);

	btod_random<4> randr;
	randr.perform(bta);
	
	tensor_t ta(bta.get_bis().get_dims());
	tod_btconv<4> conv(bta);
	conv.perform(ta);

//	std::cout << "Random tensor (dims: " << ta.get_dims() << ")" << std::endl; 
//	tensor_ctrl_t tctrl(ta);
//	const double* tptr=tctrl.req_const_dataptr();
//	size_t cnt=0;
//	for (size_t i=0; i<4; i++ ) {
//		for (size_t k=0; k<20; k++ ) std::cout << "---------";  
//		std::cout << std::endl;
//		for (size_t k=0; k<20; k++ ) std::cout << "---------";  
//		std::cout << std::endl;
//		if ( i==2 ) {
//			for (size_t k=0; k<20; k++ ) std::cout << "---------";  
//			std::cout << std::endl;
//		}
//		for (size_t a=0; a<5; a++ ) {
//			if ( a==2 ) {
//				for (size_t k=0; k<20; k++ ) std::cout << "---------"; 
//				std::cout << std::endl;
//			}
//			for (size_t j=0; j<4; j++ ) { 
//				std::cout << "||";
//				if ( j==2 ) std::cout << "|";
//				for (size_t b=0; b<5; b++ ) { 
//					if ( b==2 ) std::cout << "|"; 
//					std::cout << "   " << std::fixed << tptr[cnt++];
//				}
//			}
//			std::cout << std::endl;
//		}
//	} 

	
	tensor_t tb(ta), tc(ta), td(ta);
	permutation<4> permb, permc, permd;
	permb.permute(0,2);
	permc.permute(1,3);
	permd.permute(0,2);
	permd.permute(1,3);
	
	tod_copy<4> cpyb(ta,permb,1.0);
	cpyb.perform(tb);
	compare_ref<4>::compare("btod_random_test::test_permb",ta,tb,0.0);

	tod_copy<4> cpyc(ta,permc,1.0);
	cpyc.perform(tc);
	compare_ref<4>::compare("btod_random_test::test_permb",ta,tc,0.0);

	tod_copy<4> cpyd(ta,permd,1.0);
	cpyd.perform(td);
	compare_ref<4>::compare("btod_random_test::test_permb",ta,td,0.0);
	
	} catch(exception &exc) {
		fail_test("btod_random_test", __FILE__, __LINE__, exc.what());
	}
}


} // namespace libtensor
