#include <sstream>
#include <libtensor/dense_tensor/to_contract2_perms.h>
#include "to_contract2_perms_test.h"

namespace libtensor {


void to_contract2_perms_test::perform() throw(libtest::test_exception) {

    test_i_pi_p(1, 1);

    test_ij_i_j(1, 1);
    test_ij_i_j(5, 7);

    test_ij_j_i(1, 1);
    test_ij_j_i(5, 7);


    test_ij_ik_jk(1, 1, 1);
    test_ij_ik_jk(5, 6, 7);

    test_ijk_ij_k(1, 1, 1);
    test_ijk_ij_k(5, 6, 7);
    test_ijk_ji_k(5, 6, 7);

    test_ijk_jil_kl(5, 6, 70, 2);
    test_ijk_jil_kl(5, 6, 7, 20);

    test_ijk_pik_pj(5, 6, 70, 2);
    test_ijk_pik_pj(5, 6, 7, 20);

    test_ijab_ijkl_klab(5, 6, 7, 4, 6, 4);
    test_ijab_lijk_klab(5, 6, 7, 4, 20, 30);
}


void to_contract2_perms_test::test_i_pi_p(size_t ni, size_t np)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_i_pi_p(" << ni << ", " << np << ")";
    std::string tn = tnss.str();

    try {

    const size_t ordera =2;
    const size_t orderb =1;
    const size_t orderc =1;
/*
    index<ordera> ia1, ia2;
    ia2[0] = np - 1; ia2[1] = ni - 1;
    dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));

    index<orderb> ib1, ib2;
    ib2[0] = np - 1;
    dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

    index<orderc> ic1, ic2;
    ic2[0] = ni - 1;
    dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));//*/

    index<2> ia1, ia2; ia2[0] = np - 1; ia2[1] = ni - 1;
    index<1> ib1, ib2; ib2[0] = np - 1;
    index<1> ic1, ic2; ic2[0] = ni - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));
    dimensions<1> dimsb(index_range<1>(ib1, ib2));
    dimensions<1> dimsc(index_range<1>(ic1, ic2));

    contraction2<1, 0, 1> contr;
    contr.contract(0, 0);

    to_contract2_perms<1, 0, 1> tocp(contr, dimsa, dimsb, dimsc);

    permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
    permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
    permutation<orderc> permc; //!< Permutation of the output %tensor (c)



//    permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//    for (int i = 0; i < ordera; i++) {
//    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//    }

//    permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//      for (int i = 0; i < orderc; i++) {
//      	std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//      }

    if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ij_i_j(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ij_i_j(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    try {

    const size_t ordera =1;
    const size_t orderb =1;
    const size_t orderc =2;

    index<ordera> ia1, ia2;
    ia2[0] = ni - 1;
    dimensions<1> dimsa(index_range<1>(ia1, ia2));

    index<orderb> ib1, ib2;
    ib2[0] = nj - 1;
    dimensions<1> dimsb(index_range<1>(ib1, ib2));

    index<orderc> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsc(index_range<2>(ic1, ic2));

    contraction2<1, 1, 0> contr;

    to_contract2_perms<1, 1, 0> tocp(contr, dimsa, dimsb, dimsc);

    permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
    permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
    permutation<orderc> permc; //!< Permutation of the output %tensor (c)


//    permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//    for (int i = 0; i < ordera; i++) {
//    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//    }

//    permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//      for (int i = 0; i < orderc; i++) {
//      	std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//      }

    if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ij_j_i(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ij_j_i(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =1;
        const size_t orderb =1;
        const size_t orderc =2;


    index<ordera> ia1, ia2;
    ia2[0] = nj - 1;
    dimensions<1> dimsa(index_range<1>(ia1, ia2));

    index<orderb> ib1, ib2;
    ib2[0] = ni - 1;
    dimensions<1> dimsb(index_range<1>(ib1, ib2));

    index<orderc> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsc(index_range<2>(ic1, ic2));

    permutation<2> permc_in;
    permc_in.permute(0, 1);
    contraction2<1, 1, 0> contr(permc_in);

    to_contract2_perms<1, 1, 0> tocp(contr, dimsa, dimsb, dimsc);
    permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
    permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
    permutation<orderc> permc; //!< Permutation of the output %tensor (c)

    if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }


    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void to_contract2_perms_test::test_ij_ik_jk(size_t ni, size_t nj, size_t nk)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ij_ik_jk(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =2;
        const size_t orderb =2;
        const size_t orderc =2;


    index<ordera> ia1, ia2;
    ia2[0] = ni - 1; ia2[1] = nk - 1;
    dimensions<2> dimsa(index_range<2>(ia1, ia2));

    index<orderb> ib1, ib2;
    ib2[0] = nj - 1; ib2[1] = nk - 1;
    dimensions<2> dimsb(index_range<2>(ib1, ib2));

    index<orderc> ic1, ic2;
    ic2[0] = ni - 1; ic2[1] = nj - 1;
    dimensions<2> dimsc(index_range<2>(ic1, ic2));

    contraction2<1, 1, 1> contr;
    contr.contract(1, 1);

    to_contract2_perms<1, 1, 1> tocp(contr, dimsa, dimsb, dimsc);
    permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
    permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
    permutation<orderc> permc; //!< Permutation of the output %tensor (c)

    if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
        fail_test(tn.c_str(), __FILE__, __LINE__, "Bad dimsc.");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ijk_ij_k(size_t ni, size_t nj, size_t nk)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ijk_ij_k(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =2;
        const size_t orderb =1;
        const size_t orderc =3;


		index<ordera> ia1, ia2;
		ia2[0] = ni - 1; ia2[1] = nj - 1;
		dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));

		index<orderb> ib1, ib2;
		ib2[0] = nk - 1;
		dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

		index<orderc> ic1, ic2;
		ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
		dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));

		contraction2<2, 1, 0> contr;

		to_contract2_perms<2, 1, 0> tocp(contr, dimsa, dimsb, dimsc);
		permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
		permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
		permutation<orderc> permc; //!< Permutation of the output %tensor (c)

		if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
			fail_test(tn.c_str(), __FILE__, __LINE__, "Bad perms.");
		}

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ijk_ji_k(size_t ni, size_t nj, size_t nk)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ijk_ji_k(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =2;
        const size_t orderb =1;
        const size_t orderc =3;


		index<ordera> ia1, ia2;
		ia2[0] = nj - 1; ia2[1] = ni - 1;
		dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));

		index<orderb> ib1, ib2;
		ib2[0] = nk - 1;
		dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

		index<orderc> ic1, ic2;
		ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
		dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));

//		contraction2<2, 1, 0> contr;

	    permutation<3> permc_in;
	    permc_in.permute(0, 1);
	    contraction2<2, 1, 0> contr(permc_in);

		to_contract2_perms<2, 1, 0> tocp(contr, dimsa, dimsb, dimsc);
		permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
		permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
		permutation<orderc> permc; //!< Permutation of the output %tensor (c)
		perma.permute(0,1);

//	permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//		    for (int i = 0; i < ordera; i++) {
//		    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//		    }
//		permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//		for (int i = 0; i < orderc; i++) {
//		      	std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//		}

		if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
			fail_test(tn.c_str(), __FILE__, __LINE__, "Bad perms.");
		}

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ijk_jil_kl(size_t ni, size_t nj, size_t nk, size_t nl)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ijk_jil_kl(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =3;
        const size_t orderb =2;
        const size_t orderc =3;


		index<ordera> ia1, ia2;
		ia2[0] = nj - 1; ia2[1] = ni - 1; ia2[2] = nl - 1;
		dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));


		index<orderb> ib1, ib2;
		ib2[0] = nk - 1; ib2[1] = nl - 1;
		dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

		index<orderc> ic1, ic2;
		ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
		dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));

	    permutation<3> permc_in;
	    permc_in.permute(0, 1);
	    contraction2<2, 1, 1> contr(permc_in);
	    contr.contract(2,1);

//		std::cout << "Test0 ";
		to_contract2_perms<2, 1, 1> tocp(contr, dimsa, dimsb, dimsc);
		permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
		permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
		permutation<orderc> permc; //!< Permutation of the output %tensor (c)
//		std::cout << "Test1";

		if ( nk > nl)
			perma.permute(0,1);
		else
			permc.permute(0,1);

//	permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//		    for (int i = 0; i < ordera; i++) {
//		    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//		    }
//		permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//		for (int i = 0; i < orderc; i++) {
//		      	std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//		}
//			std::cout << "Test2";

		if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
			fail_test(tn.c_str(), __FILE__, __LINE__, "Bad perms.");
		}

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ijk_pik_pj(size_t ni, size_t nj, size_t nk, size_t np)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ijk_pik_pj(" << ni << ", " << nj << ", "
        << nk << ", " << np << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =3;
        const size_t orderb =2;
        const size_t orderc =3;


		index<ordera> ia1, ia2;
		ia2[0] = np - 1; ia2[1] = ni - 1; ia2[2] = nk - 1;
		dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));


		index<orderb> ib1, ib2;
		ib2[0] = np - 1; ib2[1] = nj - 1;
		dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

		index<orderc> ic1, ic2;
		ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = nk - 1;
		dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));

	    permutation<3> permc_in;
	    permc_in.permute(1, 2);
	    contraction2<2, 1, 1> contr(permc_in);
	    contr.contract(0,0);

//		std::cout << "Test0 ";
		to_contract2_perms<2, 1, 1> tocp(contr, dimsa, dimsb, dimsc);
		permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
		permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
		permutation<orderc> permc; //!< Permutation of the output %tensor (c)
//		std::cout << "Test1";

			permc.permute(0,1);

//	permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//		    for (int i = 0; i < ordera; i++) {
//		    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//		    }
//		permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//		for (int i = 0; i < orderc; i++) {
//		      	std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//		}
//			std::cout << "Test2";

		if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
			fail_test(tn.c_str(), __FILE__, __LINE__, "Bad perms.");
		}

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ijab_ijkl_klab(size_t ni, size_t nj, size_t nk, size_t nl, size_t na, size_t nb)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ijab_jikl_klab(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ", " << na << ", " << nb << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =4;
        const size_t orderb =4;
        const size_t orderc =4;


		index<ordera> ia1, ia2;
		ia2[0] = nj - 1; ia2[1] = ni - 1; ia2[2] = nk - 1; ia2[3] = nl - 1;
		dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));


		index<orderb> ib1, ib2;
		ib2[0] = nk - 1; ib2[1] = nl - 1; ib2[2] = na - 1; ib2[3] = nb - 1;
		dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

		index<orderc> ic1, ic2;
		ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = na - 1; ic2[3] = nb - 1;
		dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));

//	    permutation<3> permc_in;
//	    permc_in.permute(0, 1);
	    contraction2<2, 2, 2> contr;
	    contr.contract(2,0);
	    contr.contract(3,1);

		to_contract2_perms<2, 2, 2> tocp(contr, dimsa, dimsb, dimsc);
		permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
		permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
		permutation<orderc> permc; //!< Permutation of the output %tensor (c)
//		std::cout << "Test1";

//		if ( nk > nl)
//			perma.permute(0,1);
//		else
//			permc.permute(0,1);

//	permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//		    for (int i = 0; i < ordera; i++) {
//		    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//		    }
//		permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//		for (int i = 0; i < orderc; i++) {
//		      	std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//		}
//			std::cout << "Test2";

		if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
			fail_test(tn.c_str(), __FILE__, __LINE__, "Bad perms.");
		}

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

void to_contract2_perms_test::test_ijab_lijk_klab(size_t ni, size_t nj, size_t nk, size_t nl, size_t na, size_t nb)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_contract2_perms_test::test_ijab_lijk_klab(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ", " << na << ", " << nb << ")";
    std::string tn = tnss.str();

    try {

        const size_t ordera =4;
        const size_t orderb =4;
        const size_t orderc =4;


		index<ordera> ia1, ia2;
		ia2[0] = nj - 1; ia2[1] = ni - 1; ia2[2] = nk - 1; ia2[3] = nl - 1;
		dimensions<ordera> dimsa(index_range<ordera>(ia1, ia2));


		index<orderb> ib1, ib2;
		ib2[0] = nk - 1; ib2[1] = nl - 1; ib2[2] = na - 1; ib2[3] = nb - 1;
		dimensions<orderb> dimsb(index_range<orderb>(ib1, ib2));

		index<orderc> ic1, ic2;
		ic2[0] = ni - 1; ic2[1] = nj - 1; ic2[2] = na - 1; ic2[3] = nb - 1;
		dimensions<orderc> dimsc(index_range<orderc>(ic1, ic2));

	    contraction2<2, 2, 2> contr;
	    contr.contract(0,1);
	    contr.contract(3,0);

		to_contract2_perms<2, 2, 2> tocp(contr, dimsa, dimsb, dimsc);
		permutation<ordera> perma; //!< Permutation of the first input %tensor (a)
		permutation<orderb> permb; //!< Permutation of the second input %tensor (b)
		permutation<orderc> permc; //!< Permutation of the output %tensor (c)
//		std::cout << "Test1";

		perma.permute(0,2).permute(0,1);
		if ( nk*nl*na*nb/4 > (ni*nj*nk*nl - ni*nj*nk*nl/4) )
			 perma.permute(2,3);
		else
		{
			permb.permute(0,1);
		}

//	permutation<ordera> perma_calc = tocp.get_perma(); //!< Permutation of the first input %tensor (a)
//		    for (int i = 0; i < ordera; i++) {
//		    	std::cout << "perma[" << i << "] = " << perma_calc[i] << "\n";
//		    }
//		permutation<orderb> permb_calc = tocp.get_permb(); //!< Permutation of the first input %tensor (a)
//		for (int i = 0; i < orderb; i++) {
//		      	std::cout << "permb[" << i << "] = " << permb_calc[i] << "\n";
//		}
//		permutation<orderc> permc_calc = tocp.get_permc(); //!< Permutation of the first input %tensor (a)
//		for (int i = 0; i < orderc; i++) {
//				std::cout << "permc[" << i << "] = " << permc_calc[i] << "\n";
//		}
//			std::cout << "Test2";

		if(!tocp.get_perma().equals(perma) || !tocp.get_permb().equals(permb) || !tocp.get_permc().equals(permc)) {
			fail_test(tn.c_str(), __FILE__, __LINE__, "Bad perms.");
		}

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor

