#include <sstream>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/gen_block_tensor/addition_schedule.h>
#include <libtensor/core/scalar_transf_double.h>
#include "addition_schedule_test.h"

namespace libtensor {


void addition_schedule_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
    test_8();
}

/*
namespace addition_schedule_test_ns {


template<size_t N, typename T>
class schedule_comparator {
public:
    typedef typename addition_schedule<N, T>::schedule_node_t
        schedule_node_t;
    typedef std::list<schedule_node_t> schedule_type;

public:
    static void compare(const addition_schedule<N, T> &sch,
        const schedule_type &sch_ref) throw(std::string) {

        typename addition_schedule<N, T>::iterator isch = sch.begin();
        typename schedule_type::const_iterator iref = sch_ref.begin();
        while(iref != sch_ref.end()) {
            if(isch == sch.end()) {
                throw std::string("Premature end of schedule");
            }

            const schedule_node_t &node = sch.get_node(isch);
            const schedule_node_t &node_ref = *iref;

            if(node.cia != node_ref.cia) {
                std::ostringstream ss;
                ss << "Difference in canonical index of A: "
                    << node.cia << " vs. " << node_ref.cia
                    << " (expected)";
                throw ss.str();
            }

            if(node_ref.tier2 == 0 && node.tier2 != 0) {
                std::ostringstream ss;
                ss << "Tier 1 marker is bad at A["
                    << node.cia << "]";
                throw ss.str();
            }

            if(node_ref.tier2 != 0 && node.tier2 == 0) {
                std::ostringstream ss;
                ss << "Tier 2 marker is bad at A["
                    << node.cia << "]";
                throw ss.str();
            }

            if(node_ref.tier2 !=0 && node.tier2 != 0) {
                if(node_ref.tier2->cib != node.tier2->cib) {
                    std::ostringstream ss;
                    ss << "Bad tier2->cib at A[" << node.cia
                        << "]";
                    throw ss.str();
                }
                if(node_ref.tier2->trb != node.tier2->trb) {
                    std::ostringstream ss;
                    ss << "Bad tier2->trb at A[" << node.cia
                        << "]";
                    throw ss.str();
                }
            }

            if((node_ref.tier3 == 0) != (node.tier3 == 0)) {
                std::ostringstream ss;
                ss << "Tier 3 marker is bad at A["
                    << node.cia << "]";
                throw ss.str();
            }

            if(node_ref.tier3 != 0) {
                //~ if(node_ref.tier3->cib != node.tier3->cib) {
                    //~ std::ostringstream ss;
                    //~ ss << "Bad tier3->cib at A[" << node.cia
                        //~ << "]";
                    //~ throw ss.str();
                //~ }
                //~ if(node_ref.tier3->tra != node.tier3->tra) {
                    //~ std::ostringstream ss;
                    //~ ss << "Bad tier3->tra at A[" << node.cia
                        //~ << "]";
                    //~ throw ss.str();
                //~ }
            }

            if((node_ref.tier4 == 0) != (node.tier4 == 0)) {
                std::ostringstream ss;
                ss << "Tier 4 marker is bad at A["
                    << node.cia << "]";
                throw ss.str();
            }

            iref++;
            isch++;
        }
        if(isch != sch.end()) {
            throw std::string("Schedule is too long");
        }

    }
};


} // namespace addition_schedule_test_ns
using namespace addition_schedule_test_ns;
*/

/** \test Tests the addition schedule for Sym(A) = Sym(B) = Sym(C) = 0.
        Order-two block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_1() throw(libtest::test_exception) {

//	static const char *testname = "addition_schedule_test::test_1()";

//  typedef addition_schedule<2, double>::schedule_node_t schedule_node_t;
//
//  try {
//
//  index<2> i1, i2;
//  i2[0] = 9; i2[1] = 9;
//  dimensions<2> dims(index_range<2>(i1, i2));
//  block_index_space<2> bis(dims);
//  mask<2> m;
//  m[0] = true; m[1] = true;
//  bis.split(m, 5);
//
//  symmetry<2, double> syma(bis), symb(bis);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<2> bidims(bis.get_block_index_dims());
//  assignment_schedule<2, double> asch(syma);
//
//  //
//  //  Sym(A) = 0, Sym(B) = 0, Sym(C) = 0
//  //
//  //  [0, 0] <- B[0, 0] + A[0, 0]
//  //  [0, 1] <- B[0, 1] + A[0, 1]
//  //  [1, 0] <- B[1, 0] + A[1, 0]
//  //  [1, 1] <- B[1, 1] + A[1, 1]
//  //
//  schedule_node_t n00(0), n01(1), n10(2), n11(3);
//
//  sch_ref.push_back(n00);
//  sch_ref.push_back(n01);
//  sch_ref.push_back(n10);
//  sch_ref.push_back(n11);
//
//  addition_schedule<2, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<2, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for
        S(+)2 = Sym(A) > Sym(B) = Sym(C) = 0.
        Order-two block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_2() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_2()";

//  typedef addition_schedule<2, double>::schedule_node_t schedule_node_t;
//  typedef addition_schedule<2, double>::tier3_list_t tier3_list_t;
//  typedef addition_schedule<2, double>::tier3_node_t tier3_node_t;
//
//  try {
//
//  index<2> i1, i2;
//  i2[0] = 9; i2[1] = 9;
//  dimensions<2> dims(index_range<2>(i1, i2));
//  block_index_space<2> bis(dims);
//  mask<2> m;
//  m[0] = true; m[1] = true;
//  bis.split(m, 5);
//
//  symmetry<2, double> syma(bis), symb(bis);
//
//  permutation<2> perm10; perm10.permute(0, 1);
//  se_perm<2, double> se1(perm10, true);
//  syma.insert(se1);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<2> bidims(bis.get_block_index_dims());
//  assignment_schedule<2, double> asch(syma);
//
//  //
//  //  Sym(A) = S(+)2, Sym(B) = 0, Sym(C) = 0
//  //
//  //  [0, 0] <- B[0, 0] + A[0, 0]
//  //  [0, 1] <- B[0, 1] + A[0, 1]
//  //  [1, 0] <- B[1, 0] + P(+)(10) A[0, 1]
//  //  [1, 1] <- B[1, 1] + A[1, 1]
//  //
//  schedule_node_t n00(0), n01(1), n10(2), n11(3);
//  transf<2, double> tra10; tra10.permute(perm10);
//  n01.tier3 = new tier3_list_t;
//  n01.tier3->push_back(tier3_node_t(2, tra10));
//
//  sch_ref.push_back(n00);
//  sch_ref.push_back(n01);
//  sch_ref.push_back(n11);
//
//  addition_schedule<2, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<2, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for
        0 = Sym(C) = Sym(A) < Sym(B) = Perm(+, 01).
        Order-two block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_3() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_3()";

//  typedef addition_schedule<2, double>::schedule_node_t schedule_node_t;
//  typedef addition_schedule<2, double>::tier2_node_t tier2_node_t;
//
//  try {
//
//  index<2> i1, i2;
//  i2[0] = 9; i2[1] = 9;
//  dimensions<2> dims(index_range<2>(i1, i2));
//  block_index_space<2> bis(dims);
//  mask<2> m;
//  m[0] = true; m[1] = true;
//  bis.split(m, 5);
//
//  symmetry<2, double> syma(bis), symb(bis);
//
//  permutation<2> perm10; perm10.permute(0, 1);
//  se_perm<2, double> se1(perm10, true);
//  symb.insert(se1);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<2> bidims(bis.get_block_index_dims());
//  assignment_schedule<2, double> asch(syma);
//
//  //
//  //  Sym(A) = 0, Sym(B) = S(+)2, Sym(C) = 0
//  //
//  //  [0, 0] <- B[0, 0] + A[0, 0]
//  //  [0, 1] <- B[0, 1] + A[0, 1]
//  //  [1, 0] <- P+(10) B[0, 1] + A[1, 0]
//  //  [1, 1] <- B[1, 1] + A[1, 1]
//  //
//  schedule_node_t n00(0), n01(1), n10(2), n11(3);
//  transf<2, double> trb10; trb10.permute(perm10);
//  n10.tier2 = new tier2_node_t(1, trb10);
//
//  sch_ref.push_back(n00);
//  sch_ref.push_back(n01);
//  sch_ref.push_back(n10);
//  sch_ref.push_back(n11);
//
//  addition_schedule<2, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<2, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for Sym(A) > Sym(C) < Sym(B),
        Sym(A) = S(+)2, Sym(B) = S(-)2, Sym(C) = 0.
        Order-two block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_4() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_4()";

//  typedef addition_schedule<2, double>::schedule_node_t schedule_node_t;
//  typedef addition_schedule<2, double>::tier4_list_t tier4_list_t;
//  typedef addition_schedule<2, double>::tier4_node_t tier4_node_t;
//
//  try {
//
//  index<2> i1, i2;
//  i2[0] = 9; i2[1] = 9;
//  dimensions<2> dims(index_range<2>(i1, i2));
//  block_index_space<2> bis(dims);
//  mask<2> m;
//  m[0] = true; m[1] = true;
//  bis.split(m, 5);
//
//  symmetry<2, double> syma(bis), symb(bis);
//
//  permutation<2> perm10; perm10.permute(0, 1);
//  se_perm<2, double> se1(perm10, true);
//  se_perm<2, double> se2(perm10, false);
//  syma.insert(se1);
//  symb.insert(se2);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<2> bidims(bis.get_block_index_dims());
//  assignment_schedule<2, double> asch(syma);
//
//  //
//  //  Sym(A) = S(+)2, Sym(B) = S(-)2, Sym(C) = 0
//  //
//  //  [0, 0] <- B[0, 0] + A[0, 0]
//  //  [0, 1] <- B[0, 1] + A[0, 1]
//  //  [1, 0] <- P-(10) B[0, 1] + P+(10) A[0, 1]
//  //  [1, 1] <- B[1, 1] + A[1, 1]
//  //
//  schedule_node_t n00(0), n01(1), n11(3);
//  transf<2, double> tra10; tra10.permute(perm10);
//  transf<2, double> trb10; trb10.permute(perm10); trb10.scale(-1.0);
//  n01.tier4 = new tier4_list_t;
//  n01.tier4->push_back(tier4_node_t(1, 2, tra10, trb10));
//
//  sch_ref.push_back(n00);
//  sch_ref.push_back(n01);
//  sch_ref.push_back(n11);
//
//  addition_schedule<2, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<2, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for Sym(A) = Sym(B) = Sym(C) = S4.
        Order-four block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_5() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_5()";

//  typedef addition_schedule<4, double>::schedule_node_t schedule_node_t;
//
//  try {
//
//  index<4> i1, i2;
//  i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
//  dimensions<4> dims(index_range<4>(i1, i2));
//  block_index_space<4> bis(dims);
//  mask<4> m;
//  m[0] = true; m[1] = true; m[2] = true; m[3] = true;
//  bis.split(m, 5);
//
//  symmetry<4, double> syma(bis), symb(bis);
//
//  permutation<4> perm1230, perm1023;
//  perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
//  perm1023.permute(0, 1);
//  se_perm<4, double> se1(perm1230, true);
//  se_perm<4, double> se2(perm1023, true);
//  syma.insert(se1);
//  syma.insert(se2);
//  symb.insert(se1);
//  symb.insert(se2);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<4> bidims(bis.get_block_index_dims());
//  assignment_schedule<4, double> asch(bidims);
//  orbit_list<4, double> ola(syma);
//  for(orbit_list<4, double>::iterator ioa = ola.begin(); ioa != ola.end();
//      ioa++) {
//
//      abs_index<4> ai(ola.get_index(ioa), bidims);
//      asch.insert(ai.get_index());
//
//      sch_ref.push_back(schedule_node_t(ai.get_abs_index()));
//  }
//
//  addition_schedule<4, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<4, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for
        S(+)4 = Sym(A) > Sym(B) = Sym(C) = S(+)2 * S(+)2.
        Order-four block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_6() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_6()";

//  typedef addition_schedule<4, double>::schedule_node_t schedule_node_t;
//
//  try {
//
//  index<4> i1, i2;
//  i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
//  dimensions<4> dims(index_range<4>(i1, i2));
//  block_index_space<4> bis(dims);
//  mask<4> m;
//  m[0] = true; m[1] = true; m[2] = true; m[3] = true;
//  bis.split(m, 5);
//
//  symmetry<4, double> syma(bis), symb(bis);
//
//  permutation<4> perm0132, perm1230, perm1023;
//  perm0132.permute(2, 3);
//  perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
//  perm1023.permute(0, 1);
//  se_perm<4, double> se1(perm1230, true);
//  se_perm<4, double> se2(perm1023, true);
//  se_perm<4, double> se3(perm0132, true);
//  syma.insert(se1);
//  syma.insert(se2);
//  symb.insert(se2);
//  symb.insert(se3);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<4> bidims(bis.get_block_index_dims());
//  assignment_schedule<4, double> asch(bidims);
//  orbit_list<4, double> ola(syma);
//  for(orbit_list<4, double>::iterator ioa = ola.begin(); ioa != ola.end();
//      ioa++) {
//
//      abs_index<4> ai(ola.get_index(ioa), bidims);
//      asch.insert(ai.get_index());
//
//      sch_ref.push_back(schedule_node_t(ai.get_abs_index()));
//  }
//
//  addition_schedule<4, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<4, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for
        S(+)4 = Sym(B) > Sym(A) = Sym(C) = S(+)2 * S(+)2.
        Order-four block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_7() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_7()";

//  typedef addition_schedule<4, double>::schedule_node_t schedule_node_t;
//
//  try {
//
//  index<4> i1, i2;
//  i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
//  dimensions<4> dims(index_range<4>(i1, i2));
//  block_index_space<4> bis(dims);
//  mask<4> m;
//  m[0] = true; m[1] = true; m[2] = true; m[3] = true;
//  bis.split(m, 5);
//
//  symmetry<4, double> syma(bis), symb(bis);
//
//  permutation<4> perm0132, perm1230, perm1023;
//  perm0132.permute(2, 3);
//  perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
//  perm1023.permute(0, 1);
//  se_perm<4, double> se1(perm1230, true);
//  se_perm<4, double> se2(perm1023, true);
//  se_perm<4, double> se3(perm0132, true);
//  symb.insert(se1);
//  symb.insert(se2);
//  syma.insert(se2);
//  syma.insert(se3);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<4> bidims(bis.get_block_index_dims());
//  assignment_schedule<4, double> asch(bidims);
//  orbit_list<4, double> ola(syma);
//  for(orbit_list<4, double>::iterator ioa = ola.begin(); ioa != ola.end();
//      ioa++) {
//
//      abs_index<4> ai(ola.get_index(ioa), bidims);
//      asch.insert(ai.get_index());
//
//      sch_ref.push_back(schedule_node_t(ai.get_abs_index()));
//  }
//
//  addition_schedule<4, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<4, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


/** \test Tests the addition schedule for
        S(+)4 = Sym(B) > Sym(C) = A4 < Sym(A) = S(-)4.
        Order-four block tensors with two blocks along each dimension.
 **/
void addition_schedule_test::test_8() throw(libtest::test_exception) {

//    static const char *testname = "addition_schedule_test::test_8()";

//  typedef addition_schedule<4, double>::schedule_node_t schedule_node_t;
//
//  try {
//
//  index<4> i1, i2;
//  i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
//  dimensions<4> dims(index_range<4>(i1, i2));
//  block_index_space<4> bis(dims);
//  mask<4> m;
//  m[0] = true; m[1] = true; m[2] = true; m[3] = true;
//  bis.split(m, 5);
//
//  symmetry<4, double> syma(bis), symb(bis);
//
//  permutation<4> perm0132, perm1230, perm1023;
//  perm0132.permute(2, 3);
//  perm1230.permute(0, 1).permute(1, 2).permute(2, 3);
//  perm1023.permute(0, 1);
//  se_perm<4, double> se1(perm1230, false);
//  se_perm<4, double> se2(perm1023, false);
//  se_perm<4, double> se3(perm1230, true);
//  se_perm<4, double> se4(perm1023, true);
//  syma.insert(se1);
//  syma.insert(se2);
//  symb.insert(se3);
//  symb.insert(se4);
//
//  std::list<schedule_node_t> sch_ref;
//
//  dimensions<4> bidims(bis.get_block_index_dims());
//  assignment_schedule<4, double> asch(bidims);
//  orbit_list<4, double> ola(syma);
//  for(orbit_list<4, double>::iterator ioa = ola.begin(); ioa != ola.end();
//      ioa++) {
//
//      abs_index<4> ai(ola.get_index(ioa), bidims);
//      asch.insert(ai.get_index());
//
//      sch_ref.push_back(schedule_node_t(ai.get_abs_index()));
//  }
//
//  addition_schedule<4, double> sch(syma, symb);
//  sch.build(asch);
//
//  try {
//      schedule_comparator<4, double>::compare(sch, sch_ref);
//  } catch(std::string &e) {
//      fail_test(testname, __FILE__, __LINE__, e.c_str());
//  }
//
//  } catch(exception &e) {
//      fail_test(testname, __FILE__, __LINE__, e.what());
//  }
}


} // namespace libtensor
