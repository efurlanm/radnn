# Singularity example

*Last edited: 2024-06-11*

Singularity (now Apptainer) example. This Notebook runs on SDumont.

References:

- <https://github.com/LucasFernando-aes/SDumont-UserPOV/blob/main/singularity.md>
- <https://github.com/apptainer/singularity/releases/tag/v3.8.5>
- <https://apptainer.org/>
- <https://guiesbibtic.upf.edu/recerca/hpc/singularity>
- <https://pawseysc.github.io/hpc-container-training/13-singularity-intro/>
- <https://epcced.github.io/2020-12-08-Containers-Online/aio/index.html>
- <https://singularity-tutorial.github.io/02-basic-usage/>
- <https://hsf-training.github.io/hsf-training-singularity-webpage/>
- <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags>
- <https://docs.sylabs.io/guides/3.8/user-guide/>

In November 2021 the Singularity open source project joined the Linux Foundation [and was renamed](https://en.wikipedia.org/wiki/Singularity_(software)) to Apptainer.

Check for related modules:


```python
! module avail -t 2>&1 | grep -i docker
```


```python
! module avail -t 2>&1 | grep -i apptainer
```


```python
! module avail -t 2>&1 | grep -i singularity
```

    parabricks/2.5.0_singularity_sequana
    parabricks/3.0_singularity_sequana


Check the available Singularity version:


```python
! singularity --version
```

    singularity version 3.8.5-1.el7


Download example container:


```python
! singularity pull --force library://godlovedc/funny/lolcow
```

    [34mINFO:   [0m Using cached image
    [33mWARNING:[0m integrity: signature not found for object group 1
    [33mWARNING:[0m Skipping container verification



```python
! ls lolcow_latest.sif
```

    lolcow_latest.sif


The following cell ran in a terminal, starting a container and running an interactive shell inside it:
[user.name@sdumont18 container]$ singularity shell lolcow_latest.sif
Singularity> cat /etc/os-release
NAME="Ubuntu"
VERSION="16.04.5 LTS (Xenial Xerus)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 16.04.5 LTS"
VERSION_ID="16.04"
HOME_URL="http://www.ubuntu.com/"
SUPPORT_URL="http://help.ubuntu.com/"
BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"
VERSION_CODENAME=xenial
UBUNTU_CODENAME=xenial
Singularity> whoami
user.name
Singularity> hostname
sdumont18
Singularity> which cowsay
/usr/games/cowsay
Singularity> cowsay moo
 _____
< moo >
 -----
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
Singularity> fortune | cowsay | lolcat
 ______________________________________
/ A few hours grace before the madness \
\ begins again.                        /
 --------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
Singularity> exit
exit
Executing a command within the container, via the host's command line:


```python
! singularity exec lolcow_latest.sif cat /etc/os-release
```

    NAME="Ubuntu"
    VERSION="16.04.5 LTS (Xenial Xerus)"
    ID=ubuntu
    ID_LIKE=debian
    PRETTY_NAME="Ubuntu 16.04.5 LTS"
    VERSION_ID="16.04"
    HOME_URL="http://www.ubuntu.com/"
    SUPPORT_URL="http://help.ubuntu.com/"
    BUG_REPORT_URL="http://bugs.launchpad.net/ubuntu/"
    VERSION_CODENAME=xenial
    UBUNTU_CODENAME=xenial



```python
! singularity exec lolcow_latest.sif cowsay moo
```

     _____
    < moo >
     -----
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\
                    ||----w |
                    ||     ||



```python

```

Another example, downloading the PyTorch container:

<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags>

nvcr.io/nvidia/pytorch:22.03-py3


```bash
%%bash
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD=<key>
time singularity pull pytorch22.03.sif docker://nvcr.io/nvidia/pytorch:22.03-py3
```

    INFO:    Converting OCI blobs to SIF format
    INFO:    Starting build...
    Getting image source signatures
    Copying blob sha256:4d32b49e2995210e8937f0898327f196d3fcc52486f0be920e8b2d65f150a7ab
    Copying blob sha256:45893188359aca643d5918c9932da995364dc62013dfa40c075298b1baabece3
    Copying blob sha256:5ad1f2004580e415b998124ea394e9d4072a35d70968118c779f307204d6bd17
    Copying blob sha256:6ddc1d0f91833b36aac1c6f0c8cea005c87d94bab132d46cc06d9b060a81cca3
    Copying blob sha256:4cc43a803109d6e9d1fd35495cef9b1257035f5341a2db54f7a1940815b6cc65
    Copying blob sha256:e94a4481e9334ff402bf90628594f64a426672debbdfb55f1290802e52013907
    Copying blob sha256:3e7e4c9bc2b136814c20c04feb4eea2b2ecf972e20182d88759931130cfb4181
    Copying blob sha256:9463aa3f56275af97693df69478a2dc1d171f4e763ca6f7b6f370a35e605c154
    Copying blob sha256:a4a0c690bc7da07e592514dccaa26098a387e8457f69095e922b6d73f7852502
    Copying blob sha256:59d451175f6950740e26d38c322da0ef67cb59da63181eb32996f752ba8a2f17
    Copying blob sha256:eaf45e9f32d1f5a9983945a1a9f8dedbb475bc0f578337610e00b4dedec87c20
    Copying blob sha256:d8d16d6af76dc7c6b539422a25fdad5efb8ada5a8188069fcd9d113e3b783304
    Copying blob sha256:9e04bda98b05554953459b5edef7b2b14d32f1a00b979a23d04b6eb5c191e66b
    Copying blob sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
    Copying blob sha256:600169a9eeb90d73300e3d33859c1c9addf9f338bef9513e3ab719f6fbed39a4
    Copying blob sha256:5ab4a4c795873db7bd60e491a2d47ed74eefc371a3b18fa5f414e0db8a00097b
    Copying blob sha256:cc9c383bbe7ebe2c84298ba248c4a2a07063d8e691da5a7cc0e1070b7605637d
    Copying blob sha256:28f0a4457214c419a0e792989a05b32aabfe288a9a4efef8037dfb48232cb483
    Copying blob sha256:c9fa92ce1ad6b2f970cca3f68af958da9bd4d43c6962b4c91965eac822806e6f
    Copying blob sha256:6de187d8986c7f6c91fef8d21a0c79adad0ee432cb1ce5aae744a97ea8772817
    Copying blob sha256:776dd63817184de76e71037b01c4a7308fc60383b005ecb8674f40feeb340f7c
    Copying blob sha256:fbcec89863dccc0853a6330c22792801bbb3bcd5ca1a2eac1431225247aa0df0
    Copying blob sha256:cc9bd67c83a553365b3c865d6f18fb564745e1e834ea586381e00bcaf7fda0ad
    Copying blob sha256:b14fad4790ba71d322ad569201cee7d9d47061d00a6327780ab35f54e6776a7b
    Copying blob sha256:2e662c1f02bf06572c62a33873a252c6f8525b652efbb9aac0f2f7c3dd8ea69a
    Copying blob sha256:43fd87c93a0ebf98acce090ad62e17b64131c5ab423bbf1c7470a5e2556b82c2
    Copying blob sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
    Copying blob sha256:c25c11c90147bae2944f754617fb3a0a5f229ec06c0b34950ac00484883a8ca8
    Copying blob sha256:21f8a0e7678c7cbbf31db6709eaf5836b3719f4a169835d3713ddf9fbb1603a0
    Copying blob sha256:38498f1bac59aa1fb7e4b2d2d877c0e34530fc46a89c2b1bbad0bee163a87a35
    Copying blob sha256:be1dcb77f9286f979ca93c0b26303faf62fb21b2c11bf7cc18e32787ca4d5334
    Copying blob sha256:cbb8b770c5f1e4625ff08b99a18727afaf2ddf6eec76d286707c57d711f1da6f
    Copying blob sha256:570f80eab64d63e287848f4a3b9ac296e25705b1cff50b43d5952a10c67db4b5
    Copying blob sha256:3a68c3ed38b19cf1703a07884dd48edaa36092168a66158cb667601f0c0f1e75
    Copying blob sha256:44048c6af4c5854ae49eaa5d53a3d4b9d6512bd86adb2ef4b80bff57a142435f
    Copying blob sha256:aa81cbfe793ba3f5f8cb8d51d767a9c1f1751330cd0a0ecf2762174d47fe5040
    Copying blob sha256:7be5ff0e99ab9df9c87bae696e9239d7243bc3e5402dd395c3d55918dee1f657
    Copying blob sha256:fcaebb8600f0c3e3376fef56367157ef5da709a6fead68fa8c35ed42131f3791
    Copying blob sha256:5fcfa55efbe2472b0b583ddf5acea5eda40794790f423674344f8b9653cb3d90
    Copying blob sha256:30e2d0565dc130b264c65f9e95ae85ae224704dce0b7db126a5fa5c67f21f699
    Copying blob sha256:57161218948176f4f07c6b16766865e5ecfeb7448012dd916075fb94515e8aaf
    Copying blob sha256:41b60602350725f54c02464b531716dce2cfc89b49d66c44c4c53c569af61e1a
    Copying blob sha256:b46fc2bc95278510284270b1aef27fc6ae0ef55327ebfd12fc828a1999b9304c
    Copying blob sha256:218cc4dc57a5517d9e53c13465ab5d77e08b573887f398d98cce50426ae57a5d
    Copying blob sha256:312f4a0d232a53f78011e825559e0d89d68ef02d167c8221e6c96e64e2593f80
    Copying blob sha256:e4b105bef9a0766afe27b5916b8ef14766141736c58717420f59522db978d718
    Copying blob sha256:775c8d6f4c5131589a3055f9fb656c4072d07646ba713a4ff1da1a39f3585a0c
    Copying config sha256:5c38bbe3758d2bf477bbaabc1d4c93b4d9f7cef0332fc829488ccecff7024cd6
    Writing manifest to image destination
    Storing signatures
    2024/06/10 13:06:00  info unpack layer: sha256:4d32b49e2995210e8937f0898327f196d3fcc52486f0be920e8b2d65f150a7ab
    2024/06/10 13:06:01  info unpack layer: sha256:45893188359aca643d5918c9932da995364dc62013dfa40c075298b1baabece3
    2024/06/10 13:06:05  info unpack layer: sha256:5ad1f2004580e415b998124ea394e9d4072a35d70968118c779f307204d6bd17
    2024/06/10 13:06:08  info unpack layer: sha256:6ddc1d0f91833b36aac1c6f0c8cea005c87d94bab132d46cc06d9b060a81cca3
    2024/06/10 13:06:08  info unpack layer: sha256:4cc43a803109d6e9d1fd35495cef9b1257035f5341a2db54f7a1940815b6cc65
    2024/06/10 13:07:19  info unpack layer: sha256:e94a4481e9334ff402bf90628594f64a426672debbdfb55f1290802e52013907
    2024/06/10 13:07:19  info unpack layer: sha256:3e7e4c9bc2b136814c20c04feb4eea2b2ecf972e20182d88759931130cfb4181
    2024/06/10 13:07:19  info unpack layer: sha256:9463aa3f56275af97693df69478a2dc1d171f4e763ca6f7b6f370a35e605c154
    2024/06/10 13:07:19  info unpack layer: sha256:a4a0c690bc7da07e592514dccaa26098a387e8457f69095e922b6d73f7852502
    2024/06/10 13:07:19  info unpack layer: sha256:59d451175f6950740e26d38c322da0ef67cb59da63181eb32996f752ba8a2f17
    2024/06/10 13:07:23  info unpack layer: sha256:eaf45e9f32d1f5a9983945a1a9f8dedbb475bc0f578337610e00b4dedec87c20
    2024/06/10 13:07:26  info unpack layer: sha256:d8d16d6af76dc7c6b539422a25fdad5efb8ada5a8188069fcd9d113e3b783304
    2024/06/10 13:07:26  info unpack layer: sha256:9e04bda98b05554953459b5edef7b2b14d32f1a00b979a23d04b6eb5c191e66b
    2024/06/10 13:07:35  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
    2024/06/10 13:07:35  info unpack layer: sha256:600169a9eeb90d73300e3d33859c1c9addf9f338bef9513e3ab719f6fbed39a4
    2024/06/10 13:07:36  info unpack layer: sha256:5ab4a4c795873db7bd60e491a2d47ed74eefc371a3b18fa5f414e0db8a00097b
    2024/06/10 13:08:07  info unpack layer: sha256:cc9c383bbe7ebe2c84298ba248c4a2a07063d8e691da5a7cc0e1070b7605637d
    2024/06/10 13:08:08  info unpack layer: sha256:28f0a4457214c419a0e792989a05b32aabfe288a9a4efef8037dfb48232cb483
    2024/06/10 13:08:08  info unpack layer: sha256:c9fa92ce1ad6b2f970cca3f68af958da9bd4d43c6962b4c91965eac822806e6f
    2024/06/10 13:08:28  info unpack layer: sha256:6de187d8986c7f6c91fef8d21a0c79adad0ee432cb1ce5aae744a97ea8772817
    2024/06/10 13:08:29  info unpack layer: sha256:776dd63817184de76e71037b01c4a7308fc60383b005ecb8674f40feeb340f7c
    2024/06/10 13:08:35  info unpack layer: sha256:fbcec89863dccc0853a6330c22792801bbb3bcd5ca1a2eac1431225247aa0df0
    2024/06/10 13:08:36  info unpack layer: sha256:cc9bd67c83a553365b3c865d6f18fb564745e1e834ea586381e00bcaf7fda0ad
    2024/06/10 13:08:36  info unpack layer: sha256:b14fad4790ba71d322ad569201cee7d9d47061d00a6327780ab35f54e6776a7b
    2024/06/10 13:08:38  warn rootless{usr/local/nvm/versions/node/v16.6.1/bin/npm} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:08:38  warn rootless{usr/local/nvm/versions/node/v16.6.1/bin/npx} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:08:39  info unpack layer: sha256:2e662c1f02bf06572c62a33873a252c6f8525b652efbb9aac0f2f7c3dd8ea69a
    2024/06/10 13:08:39  info unpack layer: sha256:43fd87c93a0ebf98acce090ad62e17b64131c5ab423bbf1c7470a5e2556b82c2
    2024/06/10 13:08:44  info unpack layer: sha256:4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1
    2024/06/10 13:08:44  info unpack layer: sha256:c25c11c90147bae2944f754617fb3a0a5f229ec06c0b34950ac00484883a8ca8
    2024/06/10 13:08:45  info unpack layer: sha256:21f8a0e7678c7cbbf31db6709eaf5836b3719f4a169835d3713ddf9fbb1603a0
    2024/06/10 13:08:45  info unpack layer: sha256:38498f1bac59aa1fb7e4b2d2d877c0e34530fc46a89c2b1bbad0bee163a87a35
    2024/06/10 13:09:11  info unpack layer: sha256:be1dcb77f9286f979ca93c0b26303faf62fb21b2c11bf7cc18e32787ca4d5334
    2024/06/10 13:09:35  info unpack layer: sha256:cbb8b770c5f1e4625ff08b99a18727afaf2ddf6eec76d286707c57d711f1da6f
    2024/06/10 13:09:35  info unpack layer: sha256:570f80eab64d63e287848f4a3b9ac296e25705b1cff50b43d5952a10c67db4b5
    2024/06/10 13:09:35  info unpack layer: sha256:3a68c3ed38b19cf1703a07884dd48edaa36092168a66158cb667601f0c0f1e75
    2024/06/10 13:09:40  info unpack layer: sha256:44048c6af4c5854ae49eaa5d53a3d4b9d6512bd86adb2ef4b80bff57a142435f
    2024/06/10 13:09:40  info unpack layer: sha256:aa81cbfe793ba3f5f8cb8d51d767a9c1f1751330cd0a0ecf2762174d47fe5040
    2024/06/10 13:09:52  warn rootless{usr/include/rapids/libcxx/include} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow_cuda.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow_cuda.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow_dataset.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow_dataset.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow_python.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:09:53  warn rootless{usr/lib/libarrow_python.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:10:24  warn rootless{usr/lib/libparquet.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:10:24  warn rootless{usr/lib/libparquet.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 13:10:30  info unpack layer: sha256:7be5ff0e99ab9df9c87bae696e9239d7243bc3e5402dd395c3d55918dee1f657
    2024/06/10 13:10:30  info unpack layer: sha256:fcaebb8600f0c3e3376fef56367157ef5da709a6fead68fa8c35ed42131f3791
    2024/06/10 13:10:30  info unpack layer: sha256:5fcfa55efbe2472b0b583ddf5acea5eda40794790f423674344f8b9653cb3d90
    2024/06/10 13:10:31  info unpack layer: sha256:30e2d0565dc130b264c65f9e95ae85ae224704dce0b7db126a5fa5c67f21f699
    2024/06/10 13:10:31  info unpack layer: sha256:57161218948176f4f07c6b16766865e5ecfeb7448012dd916075fb94515e8aaf
    2024/06/10 13:10:32  info unpack layer: sha256:41b60602350725f54c02464b531716dce2cfc89b49d66c44c4c53c569af61e1a
    2024/06/10 13:10:32  info unpack layer: sha256:b46fc2bc95278510284270b1aef27fc6ae0ef55327ebfd12fc828a1999b9304c
    2024/06/10 13:10:33  info unpack layer: sha256:218cc4dc57a5517d9e53c13465ab5d77e08b573887f398d98cce50426ae57a5d
    2024/06/10 13:10:34  info unpack layer: sha256:312f4a0d232a53f78011e825559e0d89d68ef02d167c8221e6c96e64e2593f80
    2024/06/10 13:10:34  info unpack layer: sha256:e4b105bef9a0766afe27b5916b8ef14766141736c58717420f59522db978d718
    2024/06/10 13:10:34  info unpack layer: sha256:775c8d6f4c5131589a3055f9fb656c4072d07646ba713a4ff1da1a39f3585a0c
    INFO:    Creating SIF file...
    FATAL:   While making image from oci registry: error fetching image to cache: while building SIF from layers: while creating squashfs: create command failed: signal: killed: 
    
    real	9m1.705s
    user	36m25.881s
    sys	1m13.320s



    ---------------------------------------------------------------------------

    CalledProcessError                        Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 get_ipython().run_cell_magic('bash', '', "export SINGULARITY_DOCKER_USERNAME='$oauthtoken'\nexport SINGULARITY_DOCKER_PASSWORD=MzZ0ZmRtcWFkMXZsaWNnN2NmYTVsZTk4Nm06OTcwZGNhYjYtNjcyNy00OTQ0LTgxYzMtZDBmZmEwNDZmZWU1\ntime singularity pull pytorch22.03.sif docker://nvcr.io/nvidia/pytorch:22.03-py3\n")


    File ~/miniconda3/lib/python3.11/site-packages/IPython/core/interactiveshell.py:2517, in InteractiveShell.run_cell_magic(self, magic_name, line, cell)
       2515 with self.builtin_trap:
       2516     args = (magic_arg_s, cell)
    -> 2517     result = fn(*args, **kwargs)
       2519 # The code below prevents the output from being displayed
       2520 # when using magics with decorator @output_can_be_silenced
       2521 # when the last Python token in the expression is a ';'.
       2522 if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):


    File ~/miniconda3/lib/python3.11/site-packages/IPython/core/magics/script.py:154, in ScriptMagics._make_script_magic.<locals>.named_script_magic(line, cell)
        152 else:
        153     line = script
    --> 154 return self.shebang(line, cell)


    File ~/miniconda3/lib/python3.11/site-packages/IPython/core/magics/script.py:314, in ScriptMagics.shebang(self, line, cell)
        309 if args.raise_error and p.returncode != 0:
        310     # If we get here and p.returncode is still None, we must have
        311     # killed it but not yet seen its return code. We don't wait for it,
        312     # in case it's stuck in uninterruptible sleep. -9 = SIGKILL
        313     rc = p.returncode or -9
    --> 314     raise CalledProcessError(rc, cell)


    CalledProcessError: Command 'b"export SINGULARITY_DOCKER_USERNAME='$oauthtoken'\nexport SINGULARITY_DOCKER_PASSWORD=MzZ0ZmRtcWFkMXZsaWNnN2NmYTVsZTk4Nm06OTcwZGNhYjYtNjcyNy00OTQ0LTgxYzMtZDBmZmEwNDZmZWU1\ntime singularity pull pytorch22.03.sif docker://nvcr.io/nvidia/pytorch:22.03-py3\n"' returned non-zero exit status 255.


It is not possible to download for this container, a timeout occurs on the login node, and on an interactive node an error message appears when authenticating in the Nvidia repo. The interactive execution node is configured using:

    salloc --partition=sequana_gpu_shared --nodes=1 --ntasks=2 --time=30 --job-name test01
    ssh sdumontxxxx
    ...
    exit
    exit

The solution is to download to the local machine using Docker, save as .tar, and transfer to SDumont using scp or syncthing. On the local machine:

    docker login nvcr.io    # USERNAME&PASSWORD
    docker pull nvcr.io/nvidia/pytorch:22.03-py3
    docker save nvcr.io/nvidia/pytorch:22.03-py3  > local.tar
    transfer (to the image/ directory)

On the login node it is not possible to convert the image to SIF format:


```bash
%%bash
cd ../image
ls
time singularity build local_tar.sif docker-archive://local.tar
```

    local.tar


    INFO:    Starting build...
    Getting image source signatures
    Copying blob sha256:867d0767a47c392f80acb51572851923d6d3e55289828b0cd84a96ba342660c7
    Copying blob sha256:01e996931197162c4d65bfc6867f243df01e5f70f877c1c9beccd6978297e643
    Copying blob sha256:2ff0ade8d3c97b175cf6be4658d30fc655fa86b0198224c8f5b18e39cdd97e5f
    Copying blob sha256:fec6965e7a6b67b61a1337a169d91dcbb93ebeee67bb5d38a241f7fa4fa048ea
    Copying blob sha256:83cdade3c9b5e8313ab6d74bcfcc0492a4ae4b9fbf3d09a35ab52abaaf492d9d
    Copying blob sha256:a060c5cefec7f091c555e77e46d0130a49dfc39023365e216c154613603f456a
    Copying blob sha256:2df8c0a32afe2d088aafe59dc368643b14ddd15f9673cac769548c28ec384e04
    Copying blob sha256:899455397741dac157e85ecf485ec909e52ce1ada5d37853c93544c13e7f5f81
    Copying blob sha256:2f175b794573ab5e137bd2f5f52a2c1716bb5287ca8755ea8ad166fa9b1ee898
    Copying blob sha256:85f49f4e6923e73ae6c3ab79d4b7f1f8156141952f3f69535ca49f6a543f9e24
    Copying blob sha256:6a1014d46250ad84e998df518cecf334cf4dc3255d8da2107f5b14adb0137638
    Copying blob sha256:b9dfd77f5b0a234bacf98a6f9b5dd833dae4bd2acda78ee715a41e057669316d
    Copying blob sha256:850236713495eb5c9272fff0a67a72d80954d308cf530db88f5d238810b77530
    Copying blob sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef
    Copying blob sha256:6fb2a344ac89c7580001056daa9aaee6e145a046b503c35a9a3b5c04039ef30b
    Copying blob sha256:f89ef356505ebbde71244b0328031d581f807e9649fbc16ae9bad8c0b5044070
    Copying blob sha256:abf81ae6f4c8f107d399013c0f18ec8bd260f7b3fb25f19406102245bc0f01fe
    Copying blob sha256:852255d743c1a984be0c79a398f85c6f1c65c9ecd09ccb19ca4dd028ec34a84c
    Copying blob sha256:8fb729c89bb442004e39e13ee49ea0389bee5d241cb5e950af6160bcfd241e61
    Copying blob sha256:5ec341fc8fe7788009c830779035a302135c6df37450d0b5bd7c3fee12061d48
    Copying blob sha256:f7655918bfe60f5e7d7e95fd5887b490a251c72cf7fc070ca3bf081aee1d9b50
    Copying blob sha256:489f24d7d38164360790608a0da985125b4d69f41d63245974a9a904fc939c1e
    Copying blob sha256:1ee80d85e1cfefe604d95e7bcb45bf8cf0a117536f298cbde50da54329e679d7
    Copying blob sha256:fc3209a87194def474449b37b7ef0dae252e2452ecb5a5ffc2d81e12216005bf
    Copying blob sha256:5342e89df8e3a76605e7fe8c2a5db18fa28f757d0ab0c2fdcda0f42e9b6373c2
    Copying blob sha256:6ba71d233b759f8399a3a8a47dc829198bee3ad8eb3502c3069cbd9b53a48453
    Copying blob sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef
    Copying blob sha256:3b1792efdad9259661dfc30826f29abf456fa8b5ddac96bda338f282f5e115e1
    Copying blob sha256:3b720402b8abae528e3d207a876fed9f2c5e9de4e5f2e6f8bf46d0257710041e
    Copying blob sha256:e1aa1f9ee97ed771d635717edb0ff12757ce266966f399759e68d2d1f0d2fdab
    Copying blob sha256:7a7051e759c4f2325854116fc9b21dcfaa58f9b4aebe743ee8d78fb758c39c88
    Copying blob sha256:77a776e8014b011fc8858f05204f698998cc030966d13ae9f485a5f426040259
    Copying blob sha256:7821737d952f1296d580cb17965c84a2be8264feb3be91c967b74445c98a937b
    Copying blob sha256:14e6ddddf256545200fb3a2f8e9881624b2774359d1662b6ccdcbb2348ad364a
    Copying blob sha256:65c14c7eaf47324e28a04724c252becd2088583d39559a2c5dc946fcf296ee55
    Copying blob sha256:40f364efa84f1962030597fe2c218819b6615086fd65c653e18a53fb88c297e6
    Copying blob sha256:8456d4967bfe7d1a3f1c5812040fca36a0c07823425c278007993e552e0a9525
    Copying blob sha256:e94fa5c9c51822fa5b28c320c14ce8ee9942c59e954662a1e073acc782b9aea9
    Copying blob sha256:ea54ed1c9d39b17fabdf13617bfa0d94fa3a1691ef4a3847533f2e404e6d21fd
    Copying blob sha256:b92c6f3cf8ba79699a2607fc6010d0cc0599da479dce176332b6f03e1f2225f8
    Copying blob sha256:c68289e5466ac392537384b1d1994716e30bfe200558434067ff00c758d64b07
    Copying blob sha256:4995463ed504e50c7094525768f55bcf4c1b77e3fe720e2a350a514ad22bf904
    Copying blob sha256:80df8233699ef1706496708ee09b4f0e3681119c63cec14feac5e4b0e20f3959
    Copying blob sha256:0f010943c2be8f878e5165b682577d20ccdcefe2e5e52f873416e79648972b45
    Copying blob sha256:6d24369f9726e691bb2cae68f65dc604996d1d72194fb5e7ffbac365c8fa27cd
    Copying blob sha256:fe48bfeac91d74749aa282e68bdb462f53eb1414812c1e684a5075470c99cadf
    Copying blob sha256:9c7a2e08fe4cab75c768008df33cd4a1f5cb925bff73738cb3adc25f71a12220
    Copying config sha256:5c38bbe3758d2bf477bbaabc1d4c93b4d9f7cef0332fc829488ccecff7024cd6
    Writing manifest to image destination
    Storing signatures
    2024/06/10 20:01:34  info unpack layer: sha256:68ae9bf80e819b51d62f5e6a72de3364116f7d3013e9bb0471a66c8cb5b5c219
    2024/06/10 20:01:35  info unpack layer: sha256:bfa22944d7f27a0c3faf5dd15daa5eaf2504c8527617871539af9b67a1810c53
    2024/06/10 20:01:39  info unpack layer: sha256:845cdf20fbec7b2ab006818361f98732c2e1d3f6bd7024d0f27dafa60be3f0f5
    2024/06/10 20:01:42  info unpack layer: sha256:61018be65cafc1909fff0aa9287dfc157c8fee7766128ac959476f2598fae6c8
    2024/06/10 20:01:42  info unpack layer: sha256:1a030b7aa46d8c6802c7fc765eb42984d13dc440c3f24fe283ccc0048c70f82f
    2024/06/10 20:02:49  info unpack layer: sha256:5dac5ab6209701c30b0925443fbed56e37e0f41520022778ec37f725336df5fa
    2024/06/10 20:02:49  info unpack layer: sha256:2c7a32251a74400a58d102018097dffc8c2107fb8d0a94aa6f1c92d720200101
    2024/06/10 20:02:49  info unpack layer: sha256:ea445da495a14f1092e5919c5da43345f2cd8d215d1a2d8679892c5c34161383
    2024/06/10 20:02:49  info unpack layer: sha256:c581fefca5456c2afcbb0810d3923e2462138729c2523e54e647fd02bd993e74
    2024/06/10 20:02:49  info unpack layer: sha256:f84ad511285d7ba242bcfef9f8886d1d47f679dfc70c3566c4d7aa709b64a531
    2024/06/10 20:02:50  info unpack layer: sha256:36afb5d00374b4df1b6aa117a19b90286198ad94a595a962570eec0b1d0cd5bd
    2024/06/10 20:02:53  info unpack layer: sha256:f32148ba7fa51bd392b31c769db9060a0748a331a5b5bd597da0caa2df8bfc74
    2024/06/10 20:02:53  info unpack layer: sha256:dd54346b3b3f85aae1e4af408426b1f22ef9426cb9c1052ea20ab393e41d361e
    2024/06/10 20:03:02  info unpack layer: sha256:4ca545ee6d5db5c1170386eeb39b2ffe3bd46e5d4a73a9acbebc805f19607eb3
    2024/06/10 20:03:02  info unpack layer: sha256:89a984e0b169b985cbc658133620afb268fa70a679c3a73a85e945be0d4ff3a4
    2024/06/10 20:03:04  info unpack layer: sha256:8e0fb453a945aa36e55179ba7992ce91b437849b4408c8f825543ff0821127dc
    2024/06/10 20:03:33  info unpack layer: sha256:215a3dc3aecda761d561e9caa72cad4300755851187f58ef1fe29e09757a0e0c
    2024/06/10 20:03:33  info unpack layer: sha256:da579a8babeb09ab142bab8b3d74a42260f2ba0bfb7e0a60329c538eafaf7639
    2024/06/10 20:03:33  info unpack layer: sha256:28fce72b06e0630049d2561ebf29910b4bc7859ff36ac1d14d4dde173783a576
    2024/06/10 20:03:52  info unpack layer: sha256:7bc9557d01922b5a103b781c3921193d6c55964c14ce887820ce771e1856ff7c
    2024/06/10 20:03:53  info unpack layer: sha256:7e511e2a25bae7ae811deb0ebcb098f1d6e11d2389c60add9b56f24a5d522d20
    2024/06/10 20:03:57  info unpack layer: sha256:de4b4efd034e105d2ea4ef3e2802a0dd2f43bbd19cbf3dd445aac12e3bc8ab91
    2024/06/10 20:03:58  info unpack layer: sha256:02c9298582db401b7167704c904de045354ca94c91c824bb1e3e879bf39c08d1
    2024/06/10 20:03:58  info unpack layer: sha256:88a1a87122f69e063436e87ac8960f414f5992fc53600861e55738acf4d7bd02
    2024/06/10 20:04:01  warn rootless{usr/local/nvm/versions/node/v16.6.1/bin/npm} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:04:01  warn rootless{usr/local/nvm/versions/node/v16.6.1/bin/npx} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:04:02  info unpack layer: sha256:cf13c7a106d9d923d9f35c7c1b25aee62e296bff53f88fb9bb637998367e0c31
    2024/06/10 20:04:02  info unpack layer: sha256:6a083be126689b6eeeddccb2e3ce548c1f6521c93f1e576e3e074d26501349d4
    2024/06/10 20:04:07  info unpack layer: sha256:4ca545ee6d5db5c1170386eeb39b2ffe3bd46e5d4a73a9acbebc805f19607eb3
    2024/06/10 20:04:07  info unpack layer: sha256:78d417bf70815937d984307de91610854094c0a584a56de78ebf1af2bcaffbaf
    2024/06/10 20:04:08  info unpack layer: sha256:46ebe2a1ea63e0d198484fdf116e8506c710e9e377de4427b95083e757178cc8
    2024/06/10 20:04:08  info unpack layer: sha256:8823518a5a11ac7fe6f71076290a43eeae1904e5bcb8c1cd490219c868859ccb
    2024/06/10 20:04:34  info unpack layer: sha256:a019541ef016dc37a4d1f707f2e2c9f7a8c75912c220a710ebc63be97e6a4992
    2024/06/10 20:04:55  info unpack layer: sha256:1cf41711951dcabebe7aa7f6655637db954f0640d354be0c38585f2bd8d897dc
    2024/06/10 20:04:56  info unpack layer: sha256:e29d287a9f717bfbfe40b193fe896a9b4852d769f432205649bc62d7460df594
    2024/06/10 20:04:56  info unpack layer: sha256:42585e546c9599cb27679dfb5480ddeef45c5e6312c57524890260fbc74fd50d
    2024/06/10 20:05:00  info unpack layer: sha256:195112d6f94a8ed25a253edd5e2c383e0b0944dc4de820ece72da54461e86ef8
    2024/06/10 20:05:00  info unpack layer: sha256:76fd577178880b5ca003337b67c19c9d8de49424f7c16745738637f125345e78
    2024/06/10 20:05:14  warn rootless{usr/include/rapids/libcxx/include} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow_cuda.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow_cuda.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow_dataset.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow_dataset.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow_python.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:14  warn rootless{usr/lib/libarrow_python.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:44  warn rootless{usr/lib/libparquet.so} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:44  warn rootless{usr/lib/libparquet.so.500} ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"
    2024/06/10 20:05:50  info unpack layer: sha256:ffa4c804dc4fb2a8dae5110c44d26c886629690c9917a780903b79ba67a59bea
    2024/06/10 20:05:50  info unpack layer: sha256:d22076cedb2ab3198c0a5dc13de8dbbe8fc8f03cc5cb264ea52bb003035f7f00
    2024/06/10 20:05:50  info unpack layer: sha256:5202e6dc1a0b43a8b60e63ff139c211b8c34d9b371ef18a6d3f2bd6b4c8a7ab9
    2024/06/10 20:05:50  info unpack layer: sha256:dab1dc4f31fef0a01739ff0d683f15828e094790d29e4a2e80af77232627cfd6
    2024/06/10 20:05:51  info unpack layer: sha256:05ee3b6d570cccd2d39fdc2ef67b151d021b4681c33dde9120739a5a0e6bbe0d
    2024/06/10 20:05:52  info unpack layer: sha256:d143313679d2eb096bab646ebcbd990e029ab7c10be4237726e6af0a65eb61af
    2024/06/10 20:05:52  info unpack layer: sha256:6d542ab240499c1b8d096af6465ec2ae53cdc52ee745af4f20aa5ae61e23b2a5
    2024/06/10 20:05:52  info unpack layer: sha256:8ab66965a4e01a2a6729207e2bc6b1cf4d34655f9362c3dae5afdb3feacfdf16
    2024/06/10 20:05:55  info unpack layer: sha256:f6c7a5650dc44bd44089b477cc1910cff564f3d4ed014d02d11d611234f97162
    2024/06/10 20:05:55  info unpack layer: sha256:c95c7c5fffce9c91dbd0931737435d0b27951fa85615e308c3bb6c690cd30cfd
    2024/06/10 20:05:55  info unpack layer: sha256:cb40ba74f7e9ab9069c7262449b33ba7331a182759c712d95a4554938cd1a498
    INFO:    Creating SIF file...
    FATAL:   While performing build: while creating squashfs: create command failed: signal: killed: 



    ---------------------------------------------------------------------------

    CalledProcessError                        Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 get_ipython().run_cell_magic('bash', '', 'cd ../image\nls\nsingularity build local_tar.sif docker-archive://local.tar\n')


    File ~/miniconda3/lib/python3.11/site-packages/IPython/core/interactiveshell.py:2517, in InteractiveShell.run_cell_magic(self, magic_name, line, cell)
       2515 with self.builtin_trap:
       2516     args = (magic_arg_s, cell)
    -> 2517     result = fn(*args, **kwargs)
       2519 # The code below prevents the output from being displayed
       2520 # when using magics with decorator @output_can_be_silenced
       2521 # when the last Python token in the expression is a ';'.
       2522 if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):


    File ~/miniconda3/lib/python3.11/site-packages/IPython/core/magics/script.py:154, in ScriptMagics._make_script_magic.<locals>.named_script_magic(line, cell)
        152 else:
        153     line = script
    --> 154 return self.shebang(line, cell)


    File ~/miniconda3/lib/python3.11/site-packages/IPython/core/magics/script.py:314, in ScriptMagics.shebang(self, line, cell)
        309 if args.raise_error and p.returncode != 0:
        310     # If we get here and p.returncode is still None, we must have
        311     # killed it but not yet seen its return code. We don't wait for it,
        312     # in case it's stuck in uninterruptible sleep. -9 = SIGKILL
        313     rc = p.returncode or -9
    --> 314     raise CalledProcessError(rc, cell)


    CalledProcessError: Command 'b'cd ../image\nls\nsingularity build local_tar.sif docker-archive://local.tar\n'' returned non-zero exit status 255.


You need to copy the .tar.gz to scratch, start an interactive shell, and then convert to SIF:


```python
! cp local.tar /scratch/<user-dir>/container/
```

In a shell:

    salloc --partition=sequana_gpu_shared --nodes=1 --ntasks=2 --time=30 --job-name test01
    ssh sdumontxxxx
    cd ...
    singularity build pytorch22.03.sif docker-archive://local.tar
    ...

Running on the sdumont18:


```python
! singularity exec --nv /scratch/ampemi/<urser>/container/pytorch22.03.sif nvidia-smi
```

    Tue Jun 11 14:23:27 2024       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.6     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
    | N/A   41C    P0    38W / 250W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-PCIE...  Off  | 00000000:5E:00.0 Off |                    0 |
    | N/A   34C    P0    37W / 250W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-PCIE...  Off  | 00000000:86:00.0 Off |                    0 |
    | N/A   33C    P0    34W / 250W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-PCIE...  Off  | 00000000:AF:00.0 Off |                    0 |
    | N/A   34C    P0    35W / 250W |      0MiB / 32510MiB |      2%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


Running interactively in a shell:

    $ singularity shell --nv pytorch22.03.sif
    Singularity> nvidia-smi
    Tue Jun 11 11:15:49 2024       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.6     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla V100-PCIE...  Off  | 00000000:3B:00.0 Off |                    0 |
    | N/A   45C    P0    39W / 250W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-PCIE...  Off  | 00000000:5E:00.0 Off |                    0 |
    | N/A   36C    P0    37W / 250W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-PCIE...  Off  | 00000000:86:00.0 Off |                    0 |
    | N/A   35C    P0    35W / 250W |      0MiB / 32510MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-PCIE...  Off  | 00000000:AF:00.0 Off |                    0 |
    | N/A   36C    P0    35W / 250W |      0MiB / 32510MiB |      2%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
    Singularity> 

Check PyTorch:

```python
$ singularity shell --nv pytorch22.03.sif
Singularity> python
Python 3.8.12 | packaged by conda-forge | (default, Jan 30 2022, 23:42:07) 
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> x = torch.rand(5, 3)
>>> print(x)
tensor([[0.7518, 0.3518, 0.3545],
        [0.5821, 0.3066, 0.6131],
        [0.0897, 0.5199, 0.5225],
        [0.4321, 0.3631, 0.0693],
        [0.1293, 0.1515, 0.0600]])
>>>
>>> import torch
>>> torch.cuda.is_available()
True
```


```python

```
