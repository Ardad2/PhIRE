GENCCODE_ASSEMBLY_TYPE=-a gcc
SO=so
SOBJ=so
A=a
LIBPREFIX=lib
LIB_EXT_ORDER=.78.3
COMPILE=/home/conda/feedstock_root/build_artifacts/icu_1773821894824/_build_env/bin/aarch64-conda-linux-gnu-cc -DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -isystem /home/adadhwal/PhIRE/.mamba_candidateD_pd/include -D_REENTRANT  -DU_HAVE_ELF_H=1  -DU_ALL_IMPLEMENTATION -DU_ATTRIBUTE_DEPRECATED= -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -pipe -isystem /home/adadhwal/PhIRE/.mamba_candidateD_pd/include -fdebug-prefix-map=/home/conda/feedstock_root/build_artifacts/icu_1773821894824/work=/usr/local/src/conda/icu-78.3 -fdebug-prefix-map=/home/adadhwal/PhIRE/.mamba_candidateD_pd=/usr/local/src/conda-prefix -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -c
LIBFLAGS=-I/home/adadhwal/PhIRE/.mamba_candidateD_pd/include -DPIC -fPIC
GENLIB=/home/conda/feedstock_root/build_artifacts/icu_1773821894824/_build_env/bin/aarch64-conda-linux-gnu-cc -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O3 -pipe -isystem /home/adadhwal/PhIRE/.mamba_candidateD_pd/include -fdebug-prefix-map=/home/conda/feedstock_root/build_artifacts/icu_1773821894824/work=/usr/local/src/conda/icu-78.3 -fdebug-prefix-map=/home/adadhwal/PhIRE/.mamba_candidateD_pd=/usr/local/src/conda-prefix -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,--allow-shlib-undefined -Wl,-rpath,/home/adadhwal/PhIRE/.mamba_candidateD_pd/lib -Wl,-rpath-link,/home/adadhwal/PhIRE/.mamba_candidateD_pd/lib -L/home/adadhwal/PhIRE/.mamba_candidateD_pd/lib  -shared -Wl,-Bsymbolic
LDICUDTFLAGS=-nodefaultlibs -nostdlib
LD_SONAME=-Wl,-soname -Wl,
RPATH_FLAGS=
BIR_LDFLAGS=-Wl,-Bsymbolic
AR=aarch64-conda-linux-gnu-ar
ARFLAGS=r
RANLIB=aarch64-conda-linux-gnu-ranlib
INSTALL_CMD=/usr/bin/install -c
