#include <sstream>
#include <cstdlib>
#include <libtensor/core/allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/core/tensor_ctrl.h>
#include <libtensor/tod/tod_add.h>
#include <libtensor/tod/tod_compare.h>
#include <libtensor/tod/tod_copy.h>
#include <libtensor/tod/tod_symcontract2.h>
#include "tod_symcontract2_test.h"

namespace libtensor {

typedef tensor<2, double, std_allocator<double> > tensor2_d;
typedef tensor<4, double, std_allocator<double> > tensor4_d;

void tod_symcontract2_test::perform() throw(libtest::test_exception) {
	test_ij_ip_jp(3,8);
	test_ijab_iapq_pbqj(5,3,8,8);
}

// test this with C_{ij}=\sum_{p} A_{ip} B_{jp} - \sum_{p} A_{jp} B_{ip}
void tod_symcontract2_test::test_ij_ip_jp( size_t ni, size_t np )
	throw (libtest::test_exception)
{
	index<2> ic1, ic2;
	ic2[0]=ni-1; ic2[1]=ni-1;
	index<2> ia1, ia2;
	ia2[0]=ni-1; ia2[1]=np-1;
	index_range<2> irc(ic1, ic2);
	index_range<2> ira(ia1, ia2);
	dimensions<2> dimc(irc), dima(ira);
	tensor2_d tc(dimc), ta(dima), tb(dima), tc_ref(dimc), tc_ref2(dimc);

	{
		tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc),
			tcc_ref(tc_ref), tcc_ref2(tc_ref2);

		double* pta=tca.req_dataptr();
		double* ptb=tcb.req_dataptr();
		double* ptc=tcc.req_dataptr();
		double* ptc_ref=tcc_ref.req_dataptr();

		for (size_t i=0; i<dimc.get_size(); i++)
			ptc[i]=ptc_ref[i]=drand48();
		for (size_t i=0; i<dima.get_size(); i++) {
			pta[i]=drand48();
			ptb[i]=drand48();
		}

		tcc.ret_dataptr(ptc);

		// calculate reference
		for (size_t i=0; i<ni; i++)
		for (size_t j=0; j<ni; j++) {
			ic1[0]=i; ic1[1]=j;
			ia1[0]=i; ia2[0]=j;
			double res=0.0;
			for (size_t p=0; p<np; p++) {
				ia1[1]=p; ia2[1]=p;
				res += pta[dima.abs_index(ia1)] *
					ptb[dima.abs_index(ia2)];
			}

			ptc_ref[dimc.abs_index(ic1)]+=res;
			ic1[0]=j; ic1[1]=i;
			ptc_ref[dimc.abs_index(ic1)]-=res;
		}

		tcc_ref.ret_dataptr(ptc_ref);
		tca.ret_dataptr(pta);
		tcb.ret_dataptr(ptb);
	}

	permutation<2> pc_sym, pc;
	pc_sym.permute(0,1);

	contraction2<1,1,1> contr(pc);
	contr.contract(1,1);
	tod_symcontract2<1,1,1> op(contr,ta,tb,pc_sym,-1.0);
	op.perform(tc,1.0);

	tod_compare<2> cmp(tc,tc_ref,1e-14);

	if (! cmp.compare() ) {
		std::ostringstream str;
		str << "Result does not match reference at element "
			<< cmp.get_diff_index() << ": "
			<< cmp.get_diff_elem_1() << " (act) vs. "
			<< cmp.get_diff_elem_2() << " (ref), "
			<< cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
			<< " (diff)";

		fail_test("tod_symcontract2_test::test_ijab_iapq_pbqj()",
			__FILE__, __LINE__, str.str().c_str());
	}

	// set ta to zero again
}

// test this with C_{ijab}=\sum_{pq} A_{iapq} B_{pbqj} +/- \sum_{pq} A_{jbpq} B_{paqi}
void tod_symcontract2_test::test_ijab_iapq_pbqj( size_t na,
	size_t ni, size_t np, size_t nq )
	throw (libtest::test_exception)
{
	index<4> ic1, ic2;
	ic2[0]=ni-1; ic2[1]=ni-1; ic2[2]=na-1; ic2[3]=na-1;
	index<4> ia1, ia2;
	ia2[0]=ni-1; ia2[1]=na-1; ia2[2]=np-1; ia2[3]=nq-1;
	index<4> ib1, ib2;
	ib2[0]=np-1; ib2[1]=na-1; ib2[2]=nq-1; ib2[3]=ni-1;
	index_range<4> irc(ic1, ic2);
	index_range<4> ira(ia1, ia2);
	index_range<4> irb(ib1, ib2);
	dimensions<4> dimc(irc), dima(ira), dimb(irb);
	tensor4_d tc(dimc), ta(dima), tb(dimb), tc_ref(dimc);

	{
		tensor_ctrl<4, double> tca(ta), tcb(tb), tcc(tc),
			tcc_ref(tc_ref);

		double* pta=tca.req_dataptr();
		double* ptb=tcb.req_dataptr();
		double* ptc=tcc.req_dataptr();
		double* ptc_ref=tcc_ref.req_dataptr();

		for (size_t i=0; i<dimc.get_size(); i++)
			ptc[i]=ptc_ref[i]=drand48();
		for (size_t i=0; i<dima.get_size(); i++)
			pta[i]=drand48();
		for (size_t i=0; i<dimb.get_size(); i++)
			ptb[i]=drand48();

		tcc.ret_dataptr(ptc);

		// calculate reference
		for (size_t i=0; i<ni; i++)
		for (size_t j=0; j<ni; j++)
		for (size_t a=0; a<na; a++)
		for (size_t b=0; b<na; b++) {
			ic1[0]=i; ic1[1]=j; ic1[2]=a; ic1[3]=b;
			ia1[0]=i; ia1[1]=a;
			ib1[1]=b; ib1[3]=j;
			double res=0.0;
			for (size_t p=0; p<np; p++)
			for (size_t q=0; q<nq; q++) {
				ia1[2]=p; ia1[3]=q;
				ib1[0]=p; ib1[2]=q;
				res += pta[dima.abs_index(ia1)] *
					ptb[dimb.abs_index(ib1)];
			}

			ptc_ref[dimc.abs_index(ic1)]+=res;
			ic1[0]=j; ic1[1]=i; ic1[2]=b; ic1[3]=a;
			ptc_ref[dimc.abs_index(ic1)]-=res;
		}

		tcc_ref.ret_dataptr(ptc_ref);
		tca.ret_dataptr(pta);
		tcb.ret_dataptr(ptb);
	}

	permutation<4> pc, pc_sym;
	pc.permute(2,3);
	pc.permute(1,2);
	pc_sym.permute(0,1);
	pc_sym.permute(2,3);

	contraction2<2,2,2> contr(pc);
	contr.contract(2,0);
	contr.contract(3,2);
	tod_symcontract2<2,2,2> op(contr,ta,tb,pc_sym,-1.0);
	op.perform(tc,1.0);

	tod_compare<4> cmp(tc,tc_ref,1e-14);

	if (! cmp.compare() ) {
		std::ostringstream str;
		str << "Result does not match reference at element "
			<< cmp.get_diff_index() << ": "
			<< cmp.get_diff_elem_1() << " (act) vs. "
			<< cmp.get_diff_elem_2() << " (ref), "
			<< cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
			<< " (diff)";

		fail_test("tod_symcontract2_test::test_ijab_iapq_pbqj()",
			__FILE__, __LINE__, str.str().c_str());
	}

	// set ta to zero again
}

} // namespace libtensor

