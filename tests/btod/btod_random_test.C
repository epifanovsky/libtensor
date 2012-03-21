#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/tod/tod_btconv.h>
#include "../compare_ref.h"
#include "btod_random_test.h"

namespace libtensor {

void btod_random_test::perform() throw(libtest::test_exception)
{

	typedef std_allocator<double> allocator_t;
	typedef dense_tensor<4, double, allocator_t> tensor_t;
	typedef dense_tensor_ctrl<4, double> tensor_ctrl_t;
	typedef block_tensor<4, double, allocator_t> block_tensor_t;
	typedef block_tensor_ctrl<4, double> block_tensor_ctrl_t;

	cpu_pool cpus(1);

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

	permutation<4> perm1, perm2;
	perm1.permute(1, 3);
	perm2.permute(0, 2);
	se_perm<4, double> cycle1(perm1, true);
	se_perm<4, double> cycle2(perm2, true);

	btactrl.req_symmetry().insert(cycle1);
	btactrl.req_symmetry().insert(cycle2);

	btod_random<4> randr;
	randr.perform(bta);

	tensor_t ta(bta.get_bis().get_dims());
	tod_btconv<4> conv(bta);
	conv.perform(ta);

	tensor_t tb(ta), tc(ta), td(ta);
	permutation<4> permb, permc, permd;
	permb.permute(0,2);
	permc.permute(1,3);
	permd.permute(0,2);
	permd.permute(1,3);

	tod_copy<4>(ta, permb, 1.0).perform(cpus, true, 1.0, tb);
	compare_ref<4>::compare("btod_random_test::test_permb",ta,tb,1e-15);

	tod_copy<4>(ta, permc, 1.0).perform(cpus, true, 1.0, tc);
	compare_ref<4>::compare("btod_random_test::test_permb",ta,tc,1e-15);

	tod_copy<4>(ta, permd, 1.0).perform(cpus, true, 1.0, td);
	compare_ref<4>::compare("btod_random_test::test_permb",ta,td,1e-15);

	} catch(exception &exc) {
		fail_test("btod_random_test", __FILE__, __LINE__, exc.what());
	}
}


} // namespace libtensor
