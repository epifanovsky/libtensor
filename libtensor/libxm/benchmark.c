/*
 * Copyright (c) 2014-2016 Ilya Kaliman <ilya.kaliman@gmail.com>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "aux.h"

typedef struct setup (*make_benchmark_fn)(size_t, size_t);

struct args {
	size_t id;
	size_t size_o;
	size_t size_v;
	size_t block_size;
	int is_inmem;
};

struct setup {
	xm_dim_t dima;
	xm_dim_t dimb;
	xm_dim_t dimc;
	const char *idxa;
	const char *idxb;
	const char *idxc;
	xm_scalar_t alpha;
	xm_scalar_t beta;
	int (*init_a)(struct xm_tensor *, struct xm_allocator *, size_t, int);
	int (*init_b)(struct xm_tensor *, struct xm_allocator *, size_t, int);
	int (*init_c)(struct xm_tensor *, struct xm_allocator *, size_t, int);
};

static struct setup
make_benchmark_1(size_t o, size_t v)
{
	struct setup setup;

	setup.dima = xm_dim_4(o, o, v, v);
	setup.dimb = xm_dim_4(o, o, v, v);
	setup.dimc = xm_dim_4(o, v, o, v);
	setup.idxa = "jkbc";
	setup.idxb = "ikac";
	setup.idxc = "iajb";
	setup.init_a = xm_tensor_init_oovv;
	setup.init_b = xm_tensor_init_oovv;
	setup.init_c = xm_tensor_init_ovov;
	setup.alpha = 1.0;
	setup.beta = 1.0;

	return (setup);
}

static struct setup
make_benchmark_2(size_t o, size_t v)
{
	struct setup setup;

	setup.dima = xm_dim_4(v, v, v, v);
	setup.dimb = xm_dim_4(o, o, v, v);
	setup.dimc = xm_dim_4(o, o, v, v);
	setup.idxa = "abcd";
	setup.idxb = "ijcd";
	setup.idxc = "ijab";
	setup.init_a = xm_tensor_init_vvvv;
	setup.init_b = xm_tensor_init_oovv;
	setup.init_c = xm_tensor_init_oovv;
	setup.alpha = 1.0;
	setup.beta = 1.0;

	return (setup);
}

static struct setup
make_benchmark_3(size_t o, size_t v)
{
	struct setup setup;

	setup.dima = xm_dim_4(o, v, v, v);
	setup.dimb = xm_dim_2(o, v);
	setup.dimc = xm_dim_4(o, v, o, v);
	setup.idxa = "iabc";
	setup.idxb = "jc";
	setup.idxc = "iajb";
	setup.init_a = xm_tensor_init_ovvv;
	setup.init_b = xm_tensor_init_ov;
	setup.init_c = xm_tensor_init_ovov;
	setup.alpha = 1.0;
	setup.beta = 1.0;

	return (setup);
}

static struct setup
make_benchmark_4(size_t o, size_t v)
{
	struct setup setup;

	setup.dima = xm_dim_4(o, o, v, v);
	setup.dimb = xm_dim_4(o, v, v, v);
	setup.dimc = xm_dim_6(o, o, o, v, v, v);
	setup.idxa = "ijda";
	setup.idxb = "kdbc";
	setup.idxc = "ijkabc";
	setup.init_a = xm_tensor_init_oovv;
	setup.init_b = xm_tensor_init_ovvv;
	setup.init_c = xm_tensor_init_ooovvv;
	setup.alpha = 1.0;
	setup.beta = 0.0;

	return (setup);
}

static const make_benchmark_fn make_benchmark[] = {
	make_benchmark_1,
	make_benchmark_2,
	make_benchmark_3,
	make_benchmark_4
};

static const size_t max_benchmark_id =
    sizeof(make_benchmark) / sizeof(make_benchmark[0]);

static void
fatal(const char *msg)
{
	fprintf(stderr, "fatal error: %s\n", msg);
	abort();
}

static void
usage(void)
{
	fprintf(stderr, "usage: benchmark [-hm] [-i id] [-o no] [-v nv] "
	    "[-b block_size]\n");
	exit(1);
}

static struct args
args_default(void)
{
	struct args args;

	args.id = 2;
	args.size_o = 5;
	args.size_v = 40;
	args.block_size = 32;
	args.is_inmem = 0;

	return (args);
}

static void
args_print(const struct args *args)
{
	fprintf(stderr, "args.id=%zu\n", args->id);
	fprintf(stderr, "args.size_o=%zu\n", args->size_o);
	fprintf(stderr, "args.size_v=%zu\n", args->size_v);
	fprintf(stderr, "args.block_size=%zu\n", args->block_size);
	fprintf(stderr, "args.is_inmem=%d\n", args->is_inmem);
}

static struct args
args_parse(int argc, char **argv)
{
	struct args args;
	long arg;
	int c;

	args = args_default();

	while ((c = getopt(argc, argv, "b:c:hi:mo:v:")) != -1) {
		switch (c) {
		case 'b':
			arg = strtol(optarg, NULL, 10);
			if (arg < 1)
				usage();
			args.block_size = (size_t)arg;
			break;
		case 'i':
			arg = strtol(optarg, NULL, 10);
			if (arg < 1 || arg > (long)max_benchmark_id)
				usage();
			args.id = (size_t)arg;
			break;
		case 'm':
			args.is_inmem = 1;
			break;
		case 'o':
			arg = strtol(optarg, NULL, 10);
			if (arg < 1)
				usage();
			args.size_o = (size_t)arg;
			break;
		case 'v':
			arg = strtol(optarg, NULL, 10);
			if (arg < 1)
				usage();
			args.size_v = (size_t)arg;
			break;
		default:
			usage();
		}
	}
	return (args);
}

int
main(int argc, char **argv)
{
	struct args args;
	struct setup s;
	struct xm_allocator *allocator;
	struct xm_tensor *a, *b, *c;
	const char *path;

	xm_set_log_stream(stderr);

	args = args_parse(argc, argv);
	args_print(&args);

	s = make_benchmark[args.id-1](args.size_o, args.size_v);

	path = args.is_inmem ? NULL : "mapping";
	if ((allocator = xm_allocator_create(path)) == NULL)
		fatal("xm_allocator_create");

	if ((a = xm_tensor_create(allocator, &s.dima, "a")) == NULL)
		fatal("xm_tensor_create(a)");
	if ((b = xm_tensor_create(allocator, &s.dimb, "b")) == NULL)
		fatal("xm_tensor_create(b)");
	if ((c = xm_tensor_create(allocator, &s.dimc, "c")) == NULL)
		fatal("xm_tensor_create(c)");

	if (s.init_a(a, allocator, args.block_size, XM_INIT_RAND))
		fatal("init(a)");
	if (s.init_b(b, allocator, args.block_size, XM_INIT_RAND))
		fatal("init(b)");
	if (s.init_c(c, allocator, args.block_size, XM_INIT_ZERO))
		fatal("init(c)");

	if (xm_contract(s.alpha, a, b, s.beta, c, s.idxa, s.idxb, s.idxc))
		fatal("xm_contract");

	xm_tensor_free(a);
	xm_tensor_free(b);
	xm_tensor_free(c);
	xm_allocator_destroy(allocator);
	return (0);
}
