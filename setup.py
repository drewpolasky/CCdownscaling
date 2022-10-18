import setuptools

setuptools.setup(
	name='CCdownscaling',
	version='1.0',
	package_dir={"": "ccdown"},
	packages=setuptools.find_packages(where="ccdown"),
	url='https://github.com/drewpolasky/CCdownscaling',
	license='MIT',
	author='Drew Polasky',
	author_email='drewpolasy@gmail.com',
	description='A package providing several statistical climate downscaling tools and evaluation metrics'
)
