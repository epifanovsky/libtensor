#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/btod/btod_diagonalize.h>
#include <libtensor/btod/btod_import_raw.h>
#include <libtensor/btod/btod_tridiagonalize.h>
#include "btod_diagonalize_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_diagonalize_test::perform() throw(libtest::test_exception) {

    allocator<double>::init();

    try {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


/** \diagonalize matrix 3x3
 **/
void btod_diagonalize_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_diagonalize_test::test_1()";

    typedef allocator<double> allocator_t;

    try {
        double matrix[9] = { 1, 3, 4, 3, 2, 8, 4, 8, 3};
        //matrix with data
        libtensor::index<2> i1a, i1b;
        i1b[0] = 2; i1b[1] = 2;

        libtensor::index<1> i2a, i2b;
        i2b[0] = 2;

        dimensions<2> dims1(index_range<2>(i1a, i1b));
        dimensions<1> dims2(index_range<1>(i2a, i2b));

        block_index_space<2> bis1(dims1);
        block_index_space<1> bis2(dims2);

        block_tensor<2, double, allocator_t> bta(bis1);//input matrix
        block_tensor<2, double, allocator_t> btb(bis1);//output matrix
        block_tensor<2, double, allocator_t> btc(bis1);//output matrix
        block_tensor<2, double, allocator_t> eigvector(bis1);//input matrix
        block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
                                                    // symmetric ->tridiagonal
        block_tensor<1, double, allocator_t> eigvalue(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig_ref(bis2);//diagonal

        btod_import_raw<2>(matrix, dims1).perform(bta);
        //tridiagonalization
        btod_tridiagonalize(bta).perform(btb,S);
        btod_copy<2>(btb).perform(btc);
        //diagonalization
        btod_diagonalize diagmatrix(btc,S,1e-12,5000);
        diagmatrix.perform(btb,eigvector,eigvalue);

        //check the eigenvalues and vectors
        diagmatrix.check(bta,eigvector,eigvalue);

        //create a map for indexes in decreasing order
        size_t map[3];
        diagmatrix.sort(eigvalue,map);

        double eigens[3];
        for(size_t i =0;i<3;i++)
        {
            eigens[i] = diagmatrix.get_eigenvalue(eigvalue,map[i]);
        }
        btod_import_raw<1>(eigens, dims2).perform(eig);

        //  Prepare the reference
        double reference[3] = {12.6402,-5.5761,-1.0641};
        btod_import_raw<1>(reference, dims2).perform(eig_ref);

        //  Compare against the reference
        compare_ref<1>::compare(testname, eig, eig_ref, 1e-4);

        if(diagmatrix.get_checked()==false)
        {
            std::ostringstream ss;
            ss<<"Error! The eigenvector doesn't match the eigenvalue!";
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \diagonalize matrix 4x4 with fragmentation
 **/
void btod_diagonalize_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btod_diagonalize_test::test_2()";

    typedef allocator<double> allocator_t;

    try {
        double matrix[16] = { 4, 1, -2, 2, 1, 2, 0, 1, -2, 0, 3, -2, 2,1,-2,-1};
        //matrix with data
        libtensor::index<2> i1a, i1b;
        i1b[0] = 3; i1b[1] = 3;

        libtensor::index<1> i2a, i2b;
        i2b[0] = 3;

        dimensions<2> dims1(index_range<2>(i1a, i1b));
        dimensions<1> dims2(index_range<1>(i2a, i2b));

        block_index_space<2> bis1(dims1);
        block_index_space<1> bis2(dims2);

        mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
        mask<1> splmsk3; splmsk3[0] = true;

        bis1.split(splmsk2, 2);
        bis2.split(splmsk3, 2);

        block_tensor<2, double, allocator_t> bta(bis1);//input matrix
        block_tensor<2, double, allocator_t> btb(bis1);//output matrix
        block_tensor<2, double, allocator_t> btc(bis1);//output matrix
        block_tensor<2, double, allocator_t> eigvector(bis1);//input matrix
        block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
                                                    // symmetric ->tridiagonal
        block_tensor<1, double, allocator_t> eigvalue(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig_ref(bis2);//diagonal

        btod_import_raw<2>(matrix, dims1).perform(bta);
        //tridiagonalization
        btod_tridiagonalize(bta).perform(btb,S);
        btod_copy<2>(btb).perform(btc);
        //diagonalization
        btod_diagonalize diagmatrix(btc,S,1e-6);
        diagmatrix.perform(btb,eigvector,eigvalue);

        //check the eigenvalues and vectors
        diagmatrix.check(bta,eigvector,eigvalue);

        //create a map for indexes in decreasing order
        size_t map[4];
        diagmatrix.sort(eigvalue,map);

        double eigens[4];
        for(size_t i =0;i<4;i++)
        {
            eigens[i] = diagmatrix.get_eigenvalue(eigvalue,map[i]);
        }
        btod_import_raw<1>(eigens, dims2).perform(eig);

        //  Prepare the reference
        double reference[4] = {6.8446, 2.2685, -2.1975, 1.0844};
        btod_import_raw<1>(reference, dims2).perform(eig_ref);

        //  Compare against the reference
        compare_ref<1>::compare(testname, eig, eig_ref, 1e-4);

        if(diagmatrix.get_checked()==false)
        {
            std::ostringstream ss;
            ss<<"Error! The eigenvector doesn't match the eigenvalue!";
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \diagonalize matrix 3x3
 **/
void btod_diagonalize_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "btod_diagonalize_test::test_3()";

    typedef allocator<double> allocator_t;

    try {
        double matrix[9] = { 3, 1, 0, 1, 3, 1, 0, 1, 3};
        //matrix with data
        libtensor::index<2> i1a, i1b;
        i1b[0] = 2; i1b[1] = 2;

        libtensor::index<1> i2a, i2b;
        i2b[0] = 2;

        dimensions<2> dims1(index_range<2>(i1a, i1b));
        dimensions<1> dims2(index_range<1>(i2a, i2b));

        block_index_space<2> bis1(dims1);
        block_index_space<1> bis2(dims2);

        mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
        mask<1> splmsk3; splmsk3[0] = true;

        block_tensor<2, double, allocator_t> bta(bis1);//input matrix
        block_tensor<2, double, allocator_t> btb(bis1);//output matrix
        block_tensor<2, double, allocator_t> eigvector(bis1);//input matrix
        block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
                                                    // symmetric ->tridiagonal
        block_tensor<1, double, allocator_t> eigvalue(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig_ref(bis2);//diagonal

        btod_import_raw<2>(matrix, dims1).perform(bta);

        //diagonalization
        btod_diagonalize diagmatrix(bta,S,1e-8);

        diagmatrix.perform(btb,eigvector,eigvalue);

        //check the eigenvalues and vectors
        diagmatrix.check(bta,eigvector,eigvalue);

        //create a map for indexes in decreasing order
        size_t map[3];
        diagmatrix.sort(eigvalue,map);

        double eigens[3];
        for(size_t i =0;i<3;i++)
        {
            eigens[i] = diagmatrix.get_eigenvalue(eigvalue,map[i]);
        }
        btod_import_raw<1>(eigens, dims2).perform(eig);

        //  Prepare the reference
            double reference[3] = {4.4142, 3, 1.5858};
            btod_import_raw<1>(reference, dims2).perform(eig_ref);

        //  Compare against the reference
        compare_ref<1>::compare(testname, eig, eig_ref, 1e-4);


        if(diagmatrix.get_checked()==false)
        {
            std::ostringstream ss;
            ss<<"Error! The eigenvector doesn't match the eigenvalue!";
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \diagonalize matrix 4x4 with fragmentation
 **/
void btod_diagonalize_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "btod_diagonalize_test::test_4()";

    typedef allocator<double> allocator_t;

    try {
        double matrix[16] = { 3, 1, 0, 0,1,4,8,0,0,8,5,6,0,0,6,7};
        //matrix with data
        libtensor::index<2> i1a, i1b;
        i1b[0] = 3; i1b[1] = 3; //dimensions of a matrix - 1

        libtensor::index<1> i2a, i2b;
        i2b[0] = 3;

        dimensions<2> dims1(index_range<2>(i1a, i1b));
        dimensions<1> dims2(index_range<1>(i2a, i2b));

        block_index_space<2> bis1(dims1);
        block_index_space<1> bis2(dims2);


        mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
        mask<1> splmsk3; splmsk3[0] = true;

        bis1.split(splmsk2, 2);
        bis2.split(splmsk3, 2);

        block_tensor<2, double, allocator_t> bta(bis1);//input matrix
        block_tensor<2, double, allocator_t> btb(bis1);//output matrix
        block_tensor<2, double, allocator_t> eigvector(bis1);//input matrix
        block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
                                                    // symmetric ->tridiagonal
        block_tensor<1, double, allocator_t> eigvalue(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig_ref(bis2);//diagonal

        btod_import_raw<2>(matrix, dims1).perform(bta);

        //diagonalization
        btod_diagonalize diagmatrix(bta,S,1e-8);
        diagmatrix.perform(btb,eigvector,eigvalue);

        //check the eigenvalues and vectors
        diagmatrix.check(bta,eigvector,eigvalue);

        //create a map for indexes in decreasing order
        size_t map[4];
        diagmatrix.sort(eigvalue,map);

        double eigens[4];
        for(size_t i =0;i<4;i++)
        {
            eigens[i] = diagmatrix.get_eigenvalue(eigvalue,map[i]);
        }
        btod_import_raw<1>(eigens, dims2).perform(eig);

        //  Prepare the reference
        double reference[4] = {15.1741, 6.0222, -5.1024, 2.9061};
        btod_import_raw<1>(reference, dims2).perform(eig_ref);

        //  Compare against the reference
        compare_ref<1>::compare(testname, eig, eig_ref, 1e-4);

        if(diagmatrix.get_checked()==false)
        {
            std::ostringstream ss;
            ss<<"Error! The eigenvector doesn't match the eigenvalue!";
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \diagonalize matrix 5x5 with fragmentation
 **/
void btod_diagonalize_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "btod_diagonalize_test::test_5()";

    typedef allocator<double> allocator_t;

    try {
        double matrix[25] = { 2, -1,-1,0,0,-1,3,0,-2,0,-1,0,4,2,1,0,-2,2,8,3,0,
                0,1,3,9};
        //matrix with data
        libtensor::index<2> i1a, i1b;
        i1b[0] = 4; i1b[1] = 4;

        libtensor::index<1> i2a, i2b;
        i2b[0] = 4;

        dimensions<2> dims1(index_range<2>(i1a, i1b));
        dimensions<1> dims2(index_range<1>(i2a, i2b));

        block_index_space<2> bis1(dims1);
        block_index_space<1> bis2(dims2);

        mask<2> splmsk2; splmsk2[0] = true;splmsk2[1] = true;
        mask<1> splmsk3; splmsk3[0] = true;

        bis1.split(splmsk2, 3);
        bis2.split(splmsk3, 3);

        block_tensor<2, double, allocator_t> bta(bis1);//input matrix
        block_tensor<2, double, allocator_t> btb(bis1);//output matrix
        block_tensor<2, double, allocator_t> btc(bis1);
        block_tensor<2, double, allocator_t> eigvector(bis1);//input matrix
        block_tensor<2, double, allocator_t> S(bis1);//matrix of transformation
                                                    // symmetric ->tridiagonal
        block_tensor<1, double, allocator_t> eigvalue(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig(bis2);//diagonal
        block_tensor<1, double, allocator_t> eig_ref(bis2);//diagonal

        btod_import_raw<2>(matrix, dims1).perform(bta);

        //tridiagonalization
        btod_tridiagonalize (bta).perform(btb,S);//do it!

        btod_copy<2>(btb).perform(btc);

        //diagonalization
        btod_diagonalize diagmatrix(btc,S);// gimme diagonal matrix!

        diagmatrix.perform(btb,eigvector,eigvalue);

        //check the eigenvalues and vectors
        diagmatrix.check(bta,eigvector,eigvalue);

        //create a map for indexes in decreasing order
        size_t map[5];
        diagmatrix.sort(eigvalue,map);

        double eigens[5];
        for(size_t i =0;i<5;i++)
        {
            eigens[i] = diagmatrix.get_eigenvalue(eigvalue,map[i]);
        }
        btod_import_raw<1>(eigens, dims2).perform(eig);

        //  Prepare the reference
            double reference[5] = {12.2632, 6.2649, 4.3634, 2.0513,1.0572};
            btod_import_raw<1>(reference, dims2).perform(eig_ref);

        //  Compare against the reference
        compare_ref<1>::compare(testname, eig, eig_ref, 1e-4);

        if(diagmatrix.get_checked()==false)
        {
            std::ostringstream ss;
            ss<<"Error! The eigenvector doesn't match the eigenvalue!";
            fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
