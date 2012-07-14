#include <libtensor/core/allocator.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_print.h>
#include <libtensor/btod/btod_import_raw.h>
#include <libtensor/btod/btod_cholesky.h>
#include <libtensor/core/block_tensor.h>
#include "btod_cholesky_test.h"
#include "../compare_ref.h"
#include "libtensor/btod/btod_contract2.h"

// for our custom cholesky
#include "libtensor/btod/cholesky.h"

//#define PRINT 1

namespace libtensor {


void btod_cholesky_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5(); //this test fail for dpstrf

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


/** \cholesky Compute Cholesky decomposition of 3x3 SPD matrix
 **/
void btod_cholesky_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_cholesky_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

        //initial symmetric matrix
        double matrix[9] = { 1, 1, 1, 1, 2, 3, 1, 3, 6};
        //double matrix[9] = {1,2,3,4,5,6,7,8,9};

        index<2> i1a, i1b;
        i1b[0] = 2; i1b[1] = 2;

        dimensions<2> dims1(index_range<2>(i1a, i1b));

        
        block_index_space<2> bis1(dims1);
        
                mask<2> splmsk; splmsk[0] = true;splmsk[1] = true;
                //bis1.split(splmsk, 2);

        block_tensor<2, double, allocator_t> bta(bis1);//input matrix
        block_tensor<2, double, allocator_t> btb_ref(bis1);// reference matrix

        btod_import_raw<2>(matrix, dims1).perform(bta);

        //Decomposition
        //btod_cholesky chol(bta);
        cholesky chol(bta);

        chol.decompose();

                #ifdef PRINT
                std::cout<<std::endl;
                std::cout<<std::endl;

                std::cout<<"Rank is "<<chol.get_rank()<<std::endl;

                #endif

        //allocate the btensor for output matrix
        
        index <2> i1c;
        i1c[0] = dims1.get_dim(0) - 1;
        i1c[1] = chol.get_rank() - 1; // number of columns = rank
        
        dimensions<2> dimsb(index_range<2>(i1a,i1c));

        block_index_space<2> bisb(dimsb);

                block_tensor<2, double, allocator_t> btb(bisb);//output matrix
                block_tensor<2, double, allocator_t> btbt(bisb);//output matrix

        chol.perform(btb);

        //make matrix for comparison
        btod_copy<2>(btb).perform(btbt);
            contraction2<1,1,1> contr;
            contr.contract(1,1);
             btod_contract2<1,1,1>(contr,btb,btbt).perform(btb_ref);

                //printout the result
        #ifdef PRINT
                std::stringstream os;
                os<<std::endl;
                os<<std::endl;
                os<<testname<<std::endl;

                os<<"Input matrix is "<<std::endl;
                btod_print<2>(os).perform(bta);

                os<<"Output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb);

        os<<"Resulting output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb_ref);

                std::cout<<os.str();
        #endif
    
        //  Compare (P * L) * ( P * L)' against the input tensor
        compare_ref<2>::compare(testname, btb_ref, bta, 1e-5);
        
    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}



/**  \cholesky Compute Cholesky decomposition of 5x5 SPD matrix (with fragmentation)
 **/

void btod_cholesky_test::test_2() throw(libtest::test_exception) {

        static const char *testname = "btod_cholesky_test::test_2()";

        typedef std_allocator<double> allocator_t;

        try {
                //initial symmetric matrix

                double matrix[25] = { 1,1,1,1,1,1,2,3,4,5,1,3,6,10,15,1,4,10,20,35,
                1,5,15,35,70};

                index<2> i1a, i1b;
                i1b[0] = 4; i1b[1] = 4;

                dimensions<2> dims1(index_range<2>(i1a, i1b));

                block_index_space<2> bis1(dims1);

                //splitting
                mask<2> splmsk; splmsk[0] = true;splmsk[1] = true;
                bis1.split(splmsk, 3);

                block_tensor<2, double, allocator_t> bta(bis1);//input matrix
                block_tensor<2, double, allocator_t> btb_ref(bis1);// reference matrix

                btod_import_raw<2>(matrix, dims1).perform(bta);

                //Decomposition
                //btod_cholesky chol(bta);
        cholesky chol(bta);        

                chol.decompose();

                #ifdef PRINT
                std::cout<<std::endl;
                std::cout<<std::endl;

                std::cout<<"Rank is "<<chol.get_rank()<<std::endl;

                #endif

                //allocate the btensor for output matrix

                index <2> i1c;
                i1c[0] = dims1.get_dim(0) - 1;
                i1c[1] = chol.get_rank() - 1; // number of columns = rank

                dimensions<2> dimsb(index_range<2>(i1a,i1c));

                block_index_space<2> bisb(dimsb);

        mask<2> splmskb; splmskb[0] = true;splmskb[1] = false;
                bisb.split(splmskb, 3);

                block_tensor<2, double, allocator_t> btb(bisb);//output matrix
                block_tensor<2, double, allocator_t> btbt(bisb);//output matrix

                chol.perform(btb);


        //make matrix for comparison
                btod_copy<2>(btb).perform(btbt);
                contraction2<1,1,1> contr;
                contr.contract(1,1);
                btod_contract2<1,1,1>(contr,btb,btbt).perform(btb_ref);

                //printout the result

        #ifdef PRINT
                std::stringstream os;
                os<<std::endl;
        os<<std::endl;
        os<<testname<<std::endl;

                os<<"Input matrix is "<<std::endl;
                btod_print<2>(os).perform(bta);

                os<<"Output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb);

                os<<"Resulting output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb_ref);

                std::cout<<os.str();
        #endif

            //      Compare (P * L) * ( P * L)' against the input tensor
                compare_ref<2>::compare(testname, btb_ref, bta, 1e-5);

        //just test how well import works
        /*
        std::cout<<"Output test "<<std::endl;

                double matrixt[25] = { 1,1,1,1,1,1,2,3,4,5,1,3,6,10,15,1,4,10,20,35,
                1,5,15,35,70};

                index<2> i1at, i1bt;
                i1bt[0] = 4; i1bt[1] = 2;

                dimensions<2> dimst(index_range<2>(i1at, i1bt));

                block_index_space<2> bist(dimst);

                block_tensor<2, double, allocator_t> btt(bist);//input matrix

        btod_import_raw<2>(matrixt, btt.get_bis().get_dims()).perform(btt);

        std::stringstream ost;
        btod_print<2>(ost).perform(btt);
        std::cout<<ost.str()<<std::endl;

        //end
        */
        } catch(exception &e) {
                fail_test(testname, __FILE__, __LINE__, e.what());
        }
}



/** \cholesky 3x3 matrix, random symmetric semidefinite
**/

void btod_cholesky_test::test_3() throw(libtest::test_exception) {

        static const char *testname = "btod_cholesky_test::test_3()";

        typedef std_allocator<double> allocator_t;

        try {

                //initial symmetric matrix
                double matrix[9] = { 0.9364 , 0.9988 , 0.1483, 0.9988, 1.0956 , 0.1556 , 0.1483 , 0.1556 , 0.2988};

                //double matrix[9] = {0.431536154266377, 0.574987428745711, 0.568853251207715, 0.574987428745711, 0.902836098402728, 0.799870123313905, 0.568853251207715, 0.799870123313905, 0.762718537825839};


                index<2> i1a, i1b;
                i1b[0] = 2; i1b[1] = 2;

                dimensions<2> dims1(index_range<2>(i1a, i1b));

                block_index_space<2> bis1(dims1);

                block_tensor<2, double, allocator_t> bta(bis1);//input matrix

                block_tensor<2, double, allocator_t> btb_ref(bis1);// reference matrix

                btod_import_raw<2>(matrix, dims1).perform(bta);

                //Decomposition
                //btod_cholesky chol(bta);
        cholesky chol(bta);

                chol.decompose();

                #ifdef PRINT
                std::cout<<std::endl;
                std::cout<<std::endl;

                std::cout<<"Rank is "<<chol.get_rank()<<std::endl;

                #endif

                //allocate the btensor for output matrix

                index <2> i1c;
                i1c[0] = dims1.get_dim(0) - 1;
                i1c[1] = chol.get_rank() - 1; // number of columns = rank

                dimensions<2> dimsb(index_range<2>(i1a,i1c));

                block_index_space<2> bisb(dimsb);
                
                block_tensor<2, double, allocator_t> btb(bisb);//output matrix
                block_tensor<2, double, allocator_t> btbt(bisb);//output matrix

                chol.perform(btb);

        //make matrix for comparison
                btod_copy<2>(btb).perform(btbt);
                contraction2<1,1,1> contr;
                contr.contract(1,1);
                btod_contract2<1,1,1>(contr,btb,btbt).perform(btb_ref);

                //printout the result
        #ifdef PRINT
                std::stringstream os;
                os<<std::endl;
                os<<std::endl;
                os<<testname<<std::endl;

                os<<"Input matrix is "<<std::endl;
                btod_print<2>(os).perform(bta);

                os<<"Output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb);

                os<<"Resulting output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb_ref);

                std::cout<<os.str();
        #endif

        //      Compare (P * L) * ( P * L)' against the input tensor
                compare_ref<2>::compare(testname, btb_ref, bta, 1e-5);

        } catch(exception &e) {
                fail_test(testname, __FILE__, __LINE__, e.what());
        }
}


/**     \cholesky 5x5 matrix, random symmetric semidefinite
**/

void btod_cholesky_test::test_4() throw(libtest::test_exception) {

        static const char *testname = "btod_cholesky_test::test_4()";

        typedef std_allocator<double> allocator_t;

        try {

                //initial symmetric matrix
                double matrix[25] = {1.527119264441919, 1.243110549823781, 1.032852175317780, 1.244553509260562, 0.790231465690500,
        1.243110549823781, 1.291486160759573, 0.957613581346126, 0.964087727659145, 0.676813493606074,
        1.032852175317780, 0.957613581346126, 0.888263037016104, 0.785928335038858, 0.865950848617806,
        1.244553509260562, 0.964087727659145, 0.785928335038858, 1.697869769964709, 0.317444272791978,
        0.790231465690500, 0.676813493606074, 0.865950848617806, 0.317444272791978, 1.221196341950208
        };

                index<2> i1a, i1b;
                i1b[0] = 4; i1b[1] = 4;

                dimensions<2> dims1(index_range<2>(i1a, i1b));

                block_index_space<2> bis1(dims1);

                //splitting
                mask<2> splmsk; splmsk[0] = true;splmsk[1] = true;
                bis1.split(splmsk, 2);
        bis1.split(splmsk, 4);

                block_tensor<2, double, allocator_t> bta(bis1);//input matrix

                block_tensor<2, double, allocator_t> btb_ref(bis1);// reference matrix

                btod_import_raw<2>(matrix, dims1).perform(bta);

                //Decomposition
                //btod_cholesky chol(bta);
        cholesky chol(bta);

                chol.decompose();

                #ifdef PRINT
                std::cout<<std::endl;
                std::cout<<std::endl;

                std::cout<<"Rank is "<<chol.get_rank()<<std::endl;

                #endif

                //allocate the btensor for output matrix

                index <2> i1c;
                i1c[0] = dims1.get_dim(0) - 1;
                i1c[1] = chol.get_rank() - 1; // number of columns = rank

                dimensions<2> dimsb(index_range<2>(i1a,i1c));

                block_index_space<2> bisb(dimsb);

                mask<2> splmskb; splmskb[0] = true;splmskb[1] = false;
                bisb.split(splmskb, 2);
        bisb.split(splmskb, 4);

                
                block_tensor<2, double, allocator_t> btb(bisb);//output matrix
                block_tensor<2, double, allocator_t> btbt(bisb);//output matrix

                chol.perform(btb);

        //make matrix for comparison
                btod_copy<2>(btb).perform(btbt);
                contraction2<1,1,1> contr;
                contr.contract(1,1);
                btod_contract2<1,1,1>(contr,btb,btbt).perform(btb_ref);

                //printout the result
        #ifdef PRINT
                std::stringstream os;
                os<<std::endl;
                os<<std::endl;
                os<<testname<<std::endl;

                os<<"Input matrix is "<<std::endl;
                btod_print<2>(os).perform(bta);

                os<<"Output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb);

                os<<"Resulting output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb_ref);

                std::cout<<os.str();
        #endif
        
        //      Compare (P * L) * ( P * L)' against the input tensor
                compare_ref<2>::compare(testname, btb_ref, bta, 1e-5);

        } catch(exception &e) {
                fail_test(testname, __FILE__, __LINE__, e.what());
        }
}


/**     \cholesky 5x5 matrix, random symmetric semidefinite (for some reasons dpstrf LAPACK routine fails,whereas dpotrf is OK)
**/

void btod_cholesky_test::test_5() throw(libtest::test_exception) {

        static const char *testname = "btod_cholesky_test::test_5()";

        typedef std_allocator<double> allocator_t;

        try {

                //initial symmetric matrix
                double matrix[25] = {1.344536338296430, 1.550725623436355, 1.367493709529882, 0.918959118680545, 1.203967580379184,
        1.550725623436355, 1.974305800266248, 1.594008975456084, 1.165416041313952, 1.703687973968278,
        1.367493709529882, 1.594008975456084, 1.701536785647835, 1.309490251490619, 1.481418296209989,
        0.918959118680545, 1.165416041313952, 1.309490251490619, 1.223041559648420, 1.248452803785091,
        1.203967580379184, 1.703687973968278, 1.481418296209989, 1.248452803785091, 1.786468552187215};

                //double matrix[9] = {0.431536154266377, 0.574987428745711, 0.568853251207715, 0.574987428745711, 0.902836098402728, 0.799870123313905, 0.568853251207715, 0.799870123313905, 0.762718537825839};

                index<2> i1a, i1b;
                i1b[0] = 4; i1b[1] = 4;

                dimensions<2> dims1(index_range<2>(i1a, i1b));

                block_index_space<2> bis1(dims1);

                //splitting
                mask<2> splmsk; splmsk[0] = true;splmsk[1] = true;
                bis1.split(splmsk, 2);

                block_tensor<2, double, allocator_t> bta(bis1);//input matrix
                block_tensor<2, double, allocator_t> btb_ref(bis1);// reference matrix

                btod_import_raw<2>(matrix, dims1).perform(bta);

                //Decomposition
                //btod_cholesky chol(bta);
        cholesky chol(bta);

                chol.decompose();

                #ifdef PRINT
                std::cout<<std::endl;
                std::cout<<std::endl;

                std::cout<<"Rank is "<<chol.get_rank()<<std::endl;

                #endif

                //allocate the btensor for output matrix

                index <2> i1c;
                i1c[0] = dims1.get_dim(0) - 1;
                i1c[1] = chol.get_rank() - 1; // number of columns = rank

                dimensions<2> dimsb(index_range<2>(i1a,i1c));

                block_index_space<2> bisb(dimsb);

                mask<2> splmskb; splmskb[0] = true;splmskb[1] = false;
                bisb.split(splmskb, 2);
                
                block_tensor<2, double, allocator_t> btb(bisb);//output matrix
                block_tensor<2, double, allocator_t> btbt(bisb);//output matrix

                chol.perform(btb);

        //make matrix for comparison
                btod_copy<2>(btb).perform(btbt);
                contraction2<1,1,1> contr;
                contr.contract(1,1);
                btod_contract2<1,1,1>(contr,btb,btbt).perform(btb_ref);

                //printout the result

        #ifdef PRINT
                std::stringstream os;
                os<<std::endl;
                os<<std::endl;
                os<<testname<<std::endl;

                os<<"Input matrix is "<<std::endl;
                btod_print<2>(os).perform(bta);

                os<<"Output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb);

                os<<"Resulting output matrix is "<<std::endl;
                btod_print<2>(os).perform(btb_ref);

                std::cout<<os.str();
        #endif        

        //      Compare (P * L) * ( P * L)' against the input tensor
                compare_ref<2>::compare(testname, btb_ref, bta, 1e-5);

        } catch(exception &e) {
                fail_test(testname, __FILE__, __LINE__, e.what());
        }
}


} // namespace libtensor
