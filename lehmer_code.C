#include "lehmer_code.h"

namespace libtensor {

lehmer_code::lehmer_code() {
	size_t fact = 1;
	for(size_t i=0; i<max_tensor_order-1; i++) {
		fact *= (i+1); m_fact[i] = fact;
	}
	for(size_t i=0; i<max_tensor_order-1; i++) {
		m_codes[i].resize(m_fact[i+1], NULL);
	}
}

size_t lehmer_code::perm2code(const permutation &p) throw(exception) {
	size_t order = p.get_order();
	size_t seq[order];
	size_t code = 0;
	for(size_t i=0; i<order; i++) seq[i]=i;
	p.apply(order, seq);
	for(size_t i=0; i<order-1; i++) {
		register size_t k = seq[i];
		for(register size_t j=i+1; j<order; j++) if(seq[j]>k) seq[j]--;
		code += k*m_fact[order-i-2];
	}
	return code;
}

const permutation &lehmer_code::code2perm(const size_t order,
	const size_t code) throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(order < 2 || order > max_tensor_order) {
		throw_exc("code2perm(const size_t, const size_t)",
			"Incorrect permutation order");
	}
	if(code > m_fact[order-1]-1) {
		throw_exc("code2perm(const size_t, const size_t)",
			"Invalid code");
	}
#endif // LIBTENSOR_DEBUG
	permutation *p = (m_codes[order-2])[code];
	if(p == NULL) {
		p = new permutation(order);

		size_t c = code;
		size_t seq[order-1];
		size_t i=order-1;
		do {
			i--;
			seq[i] = c/m_fact[i];
			c = c%m_fact[i];
		} while(i != 0);

		//printf("order=%lu, code=%lu: <", order, code);
		//for(size_t j=order-1; j!=0; j--) printf("%lu", seq[j-1]);
		//printf(">\n");

		bool done = false;
		do {
			i = 0;
			while(i<order-1 && seq[i]==0) i++;
			if(i!=order-1) {
				//printf("sigma_%lu\n", order-i-2);
				p->permute(order-i-2, order-i-1);
				if(i==0) seq[i]=0;
				else { seq[i-1]=seq[i]-1; seq[i]=0; }
			} else {
				done = true;
			}
		} while(!done);
		p->invert();

		(m_codes[order-2])[code] = p;
	}
	return *p;
}

void lehmer_code::throw_exc(const char *method, const char *msg) const
	throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::lehmer_code::%s] %s.", method, msg);
	throw exception(s);
}

} // namespace libtensor

