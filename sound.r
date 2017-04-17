#MATLAB code
#for problem2 using wavio

s=sound1('<name of file>')

function s = sound1(fname)
	%
	audioinfo(fname)
	s = audioread(fname);
	s = s(2000:2000+4*55100);
	figure;
	subplot(2,1,1);
	plot(s(100000:103000))
	$
	fs = fft(a);
	fs2 = abs(fs).^2
	%
	mm = max(fs2);
	ix = find(fs2>(mm/10));
	f = ix(1);
	f0 = 44100/412160;
	subplot(2,1,2);
	plot(fa2(1:3000));
	f*f0

end