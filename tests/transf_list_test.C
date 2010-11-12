#include <sstream>
#include <libtensor/core/transf_list.h>
#include <libtensor/btod/transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include "transf_list_test.h"

namespace libtensor {


void transf_list_test::perform() throw(libtest::test_exception) {

	test_1();
	test_2();
}


namespace transf_list_test_ns {


template<size_t N>
std::ostream &operator<<(std::ostream &os, const transf<N, double> &tr) {
	os << "[" << tr.get_perm() << "; " << tr.get_coeff() << "]";
	return os;
}


template<size_t N, typename T>
std::string trlist_compare(const char *testname, const index<N> &idx,
	const transf_list<N, T> &trlist,
	const std::list< transf<N, T> > &trlist_ref) {

	typedef std::pair<transf<N, T>, bool> trpair_t;
	std::list<trpair_t> trplist_ref;
	for(typename std::list< transf<N, T> >::const_iterator i =
		trlist_ref.begin(); i != trlist_ref.end(); i++) {
		trplist_ref.push_back(trpair_t(*i, false));
	}

	size_t n = 0;
	for(typename transf_list<N, T>::iterator i = trlist.begin();
		i != trlist.end(); i++, n++) {

		typename std::list<trpair_t>::iterator iref =
			trplist_ref.begin();
		while(iref != trplist_ref.end()) {
			if(iref->first == trlist.get_transf(i)) break;
			iref++;
		}
		if(iref == trplist_ref.end()) {
			std::ostringstream ss;
			ss << "Transformation " << trlist.get_transf(i) <<
				" for " << idx
				<< " not found in reference list.";
			return ss.str();
		}
		if(iref->second) {
			std::ostringstream ss;
			ss << "Duplicate transformation "
				<< trlist.get_transf(i) << " for "
				<< idx << ".";
			return ss.str();
		}
		iref->second = true;
	}
	if(n != trplist_ref.size()) {
		std::ostringstream ss;
		ss << "Wrong number " << n << " != " << trplist_ref.size()
			<< " (ref) of transformations listed for "
			<< idx << ".";
		return ss.str();
	}
	return std::string();
}


} // namespace transf_list_test_ns
using namespace transf_list_test_ns;


/**	\brief Tests transformation lists for diagonal and non-diagonal blocks
		of a two-index tensor with empty symmetry.
 **/
void transf_list_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "transf_list_test::test_1()";

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	symmetry<2, double> sym(bis);

	//	Reference lists

	transf<2, double> trref;
	std::list< transf<2, double> > trlist00_ref, trlist01_ref;
	trlist00_ref.push_back(trref);
	trlist01_ref.push_back(trref);

	//	Make transformation lists

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	transf_list<2, double> trl00(sym, i00), trl01(sym, i01),
		trl10(sym, i10), trl11(sym, i11);

	//	Check against the reference

	std::string s;
	s = trlist_compare(testname, i00, trl00, trlist00_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
	s = trlist_compare(testname, i01, trl01, trlist01_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
	s = trlist_compare(testname, i10, trl10, trlist01_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
	s = trlist_compare(testname, i11, trl11, trlist00_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


/**	\brief Tests transformation lists for diagonal and non-diagonal blocks
		of a two-index tensor with S2(+) symmetry.
 **/
void transf_list_test::test_2() throw(libtest::test_exception) {

	static const char *testname = "transf_list_test::test_2()";

	try {

	index<2> i1, i2;
	i2[0] = 2; i2[1] = 2;
	mask<2> msk;
	msk[0] = true; msk[1] = true;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);
	bis.split(msk, 1);
	symmetry<2, double> sym(bis);

	se_perm<2, double> se1(permutation<2>().permute(0, 1), true);
	sym.insert(se1);

	//	Reference lists

	transf<2, double> trref;
	std::list< transf<2, double> > trlist00_ref, trlist01_ref;

	trref.reset();
	trlist00_ref.push_back(trref);
	trref.reset();
	trref.permute(permutation<2>().permute(0, 1));
	trlist00_ref.push_back(trref);

	trref.reset();
	trlist01_ref.push_back(trref);

	//	Make transformation lists

	index<2> i00, i01, i10, i11;
	i10[0] = 1; i01[1] = 1;
	i11[0] = 1; i11[1] = 1;
	transf_list<2, double> trl00(sym, i00), trl01(sym, i01),
		trl10(sym, i10), trl11(sym, i11);

	//	Check against the reference

	std::string s;
	s = trlist_compare(testname, i00, trl00, trlist00_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
	s = trlist_compare(testname, i01, trl01, trlist01_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
	s = trlist_compare(testname, i10, trl10, trlist01_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());
	s = trlist_compare(testname, i11, trl11, trlist00_ref);
	if(!s.empty()) fail_test(testname, __FILE__, __LINE__, s.c_str());

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
