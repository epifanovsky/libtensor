#include <iostream>
#include <sstream>
#include <mpi.h>
#include <libtensor/version.h>
#include <libtensor/linalg/linalg.h>
#include "libtensor_ctf_block_tensor_suite.h"

using namespace libtensor;
using namespace std;
using libtest::test_exception;


class suite_handler : public libtest::suite_event_handler {
private:
    bool print;

public:
    suite_handler(bool print_) : print(print_) { }

    virtual void on_suite_start(const char *suite) { }
    virtual void on_suite_end(const char *suite) { }

    virtual void on_test_start(const char *test) {
        if(!print) return;
        cout << "Test " << test << " ... ";
    }

    virtual void on_test_end_success(const char *test) {
        if(!print) return;
        cout << "done." << endl;
    }

    virtual void on_test_end_exception(const char *test,
        const test_exception &e) {

        if(!print) return;
        cout << "FAIL!" << endl << e.what() << endl;
    }

};


int main(int argc, char **argv) {

    linalg::rng_setup(0);

    int nproc, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(myid == 0) {
        ostringstream ss1, ss2;
        ss1 << " Unit tests for libtensor " << version::get_string();
        ss2 << " Cyclops Tensor Framework with " << nproc << " MPI "
            << (nproc == 1 ? "process" : "processes");
        string separator(std::max(ss1.str().size(), ss2.str().size()), '-');
        cout << separator << endl << ss1.str() << endl << ss2.str() << endl
             << separator << endl;
    }

    suite_handler handler(myid == 0);
    libtensor_ctf_block_tensor_suite suite;
    suite.set_handler(&handler);

    suite.run_all_tests();

    MPI_Finalize();
}

