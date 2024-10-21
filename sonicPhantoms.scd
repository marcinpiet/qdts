(
/*
Dependencies
- RedUniverse
- OversamplingOscillators
*/

//starts the synthesis server and loads all the necessary functions
s.waitForBoot({

	/*
	~shiftL is the left shift function. For example, ~shiftL.value([1,2,3] -> [2,3,0]

	Input: An array x of arbitrary size.

	Process:
	- If the size of x is less than 2, it returns a copy of x unchanged.
	- Otherwise, it creates a new array shifted and shifts all elements of x one position to the left, filling the last position with 0.

	Output: A new array that is a left-shifted version of x.
	*/

	~shiftL = { |x|
		var n, shifted;
		n = x.size;
		if (n < 2, { x.copy }); // If the array has less than 2 elements, return a copy as-is
		shifted = Array.fill(n, 0); // Create an array of zeros of size N
		(0..n-2).do { |i| shifted[i] = x[i+1]; }; // Shift elements left
		shifted[n-1] = 0; // Set the last element to 0
		shifted // Return the shifted array
	};

	//test
	//~shiftL.value([1,2,3]);

	/*
	~shiftR is the right shift function. For example, ~shiftR.value([1,2,3]) -> [0,1,2]

	Input: An array x.

	Process:
	1. If x has less than 2 elements, it returns a copy unchanged.
	2.Otherwise, it creates a new array shifted, shifting all elements of x one position to the right, filling the first position with 0.

	Output: A new array that is a right-shifted version of x.
	*/

	~shiftR = { |x|
		var n, shifted;
		n = x.size;
		if (n < 2, { x.copy }); // If the array has less than 2 elements, return a copy as-is
		shifted = Array.fill(n, 0); // Create an array of zeros of size n
		(1..n-1).do { |i| shifted[i] = x[i-1]; }; // Shift elements right
		shifted[0] = 0; // Set the first element to 0
		shifted; // Return the shifted array
	};

	//~shiftR.value(RedVector2D[1,2,3]);

	/*

	//apply shiftL and shiftR number of times

	Input: An array x and an integer times.

	Process:
	1.If times is less than 0, it prints an error message and returns the original array.
	2.It then iterates times number of times, applying the left shift (~shiftL) or right shift (shiftR) on x and updating the result each time.

	Output: The array x after being left-shifted times times.
	*/

	~shiftL_N = { |x, times|
		var result;
		if (times < 0, { "Invalid shift times: must be >= 0".postln; ^x });
		result = x.copy;
		times.do { result = ~shiftL.value(result); };
		result; // Ensure this returns a valid array
	};

	~shiftR_N = { |x, times|
		var result;
		if (times < 0, { "Invalid shift times: must be >= 0".postln; ^x });
		result = x.copy;
		times.do { result = ~shiftR.value(result); };
		result // Ensure this returns a valid array
	};

	//x = [1, 2, 3];
	//~shiftL_N.value(x, 1).postln;  // Expected: [2, 3, 0]
	//~shiftR_N.value(x, 1).postln;  // Expected: [0, 1, 2]

	/*
	Helper function SN - subarray extraction

	Input: An array y and an integer n.
	Process:
	Returns a subarray from index 0 to n-1 of the input array y.
	Output: A subarray of y.
	*/

	~sn = { |y, n|
		// Return subarray starting from index 0 to (n-1)
		y.copyRange(0, n-1)
	};

	/*
	Input: Two arrays array1 and array2.

	Process:
	1.First checks if both arrays have the same size. If not, throws an error.
	2.Then calculates the dot product by summing element-wise products of corresponding elements in array1 and array2.

	Output: A single scalar value, the result of the dot product.
	*/

	// Define the dot product function globally
	~dotProduct = { |array1, array2|
		var result = 0;
		if (array1.size != array2.size, {
			"Array size mismatch: array1 size = %, array2 size = %"
			.format(array1.size, array2.size)
			.postln;
			Error("Arrays must have the same size for dot product").throw;
		});
		array1.do({ |val, i|
			result = result + (val * array2[i]);
		});
		result;
	};

	//a = RedVector2D[1, 2, 3];
	//b = RedVector2D[4, 5, 6];
	//~dotProduct.value(a, b).postln; // Should return the sum of element-wise products

	//The function A generates the right part of the equation (17) in the paper considering A_1=1. For example,
	//A(2,3,4)=( 1*2 + 2*3 + 3*4
	//           1*3 + 2*4
	//           1*4             )
	//A:R^N -> R^N, where N+1 is the number of sinusoids and N is the number of target harmonics.


	//~a - Function to compute a matrix-like result based on dot products

	/*
	Input: An array x.

	Process:
	1.Creates a new array y by prepending 1 to x.
	2.For each i from 0 to n-1, it calculates the dot product between a subarray of y and itself.

	Output: An array z where each element is the result of the dot product.
	*/

	~a = { |x|
		var n = x.size;
		var y = [1] ++ x; // Create Y by prepending 1 to x
		var z = Array.fill(n, 0); // Initialize Z as an array of zeros of size n

		// Fill Z with dot products, using only the matching portion of Y
		n.do { |i|
			var sn_array = ~sn.value(y, i + 1); // Get subarray SN(y, i+1)
			z[i] = ~dotProduct.value(y.copyRange(0, i), sn_array); // Dot product with matching portion of Y
		};

		z // Return the result Z
	};

	//~a.value([2,3,4]);


	// Function F: calculates the difference between ~a(x) and t
	/*

	Input: Two arrays, x and t.

	Process:
	1.Calls ~a.value(x) to compute a result and then subtracts the array t element-wise.

	Output: The array resulting from ~a(x) - t.
	*/
	~f = { |x, t|
		var y = ~a.value(x) - t; // Apply the function A to x, then subtract t element-wise
		y // Return the result Y
	};

	// Test ~f with an input array and target array
	//t = [1, 1, 1, 1, 1]; // Target array
	//~f.value(x, t).postln; // Expected output: the difference between ~a(x) and t

	//x = [2, 3, 4]; // Input array X
	//t = [1, 1, 1]; // Input array T
	//~result = ~f.value(x, t);
	//~result.postln; // Output: result of A(x) - t

	// Jacobian matrix function using shiftL_N and shiftR_N
	/*
	Input: An array x.

	Process:
	1.Initializes a 2D array z of zeros of size n x n where n is the size of x.
	2.Computes left and right shifts on x and fills the matrix with these values, then adds 1 to the diagonal elements to form the Jacobian matrix.

	Output: A 2D array z representing the Jacobian matrix.
	*/
	~df = { |x|
		var n = x.size;
		var z = Array.fill(n, { Array.fill(n, 0) }); // Create a 2D array of zeros of size n x n
		var temp;

		(0..n-2).do { |i| // Loop over the rows
			temp = ~shiftL_N.value(x, i + 1) + ~shiftR_N.value(x, i + 1); // Get left and right shifts

			if (temp.notNil, {
				"temp[%] = %".format(i, temp).postln; // Print temp for debugging
				(0..n-1).do { |j|
					if (temp[j].notNil, {
						z[i][j] = temp[j]; // Assign temp values to the z matrix
					}, {
						"temp[%][%] is nil".format(i, j).postln;
					});
				};
			}, {
				"temp is nil at iteration %".format(i).postln;
			});
		};

		// Adding the identity matrix to Z
		(0..n-1).do { |i|
			z[i][i] = z[i][i] + 1; // Increment diagonal elements
		};

		z // Return the Jacobian matrix Z
	};

	//x = [1,2,3];
	//~df.value(x);


	//~min_func - Minimization function
	/*
	Input: An array x, an integer maxIterations, and a float tolerance.

	Process:
	1.Initializes solution as a zero-filled array.
	2.Iteratively updates the solution using left shifts and random noise while calculating a function ~a(x) and minimizing the dot product of this function.
	3.The process stops when the function reaches the specified tolerance or the maximum number of iterations.
	4.The final solution is sorted from highest to lowest.

	Output: The final sorted solution as an array.
	*/

	~min_func = { |x, maxIterations = 100, tolerance = 1e-9|
		var n = x.size;
		var solution = Array.fill(n, 0);  // Initialize solution as a zero-filled array
		var df_val, f_init, f_temp, dot_f_init, dot_f_temp;
		var temporal_solution;
		var escape = false;  // Flag for stopping condition
		var step = 0;  // Iteration counter

		"Checking sizes: x size = %".format(x.size).postln;

		// Step 1: Compute the initial value using the function a
		f_init = ~a.value(x);
		dot_f_init = ~dotProduct.value(f_init, f_init);
		"Initial dot_f_init: %".format(dot_f_init).postln;

		// Iterate over the vector x, with a maximum number of iterations
		while({
			(step < maxIterations) && (escape.not)  // Wrap condition in curly braces to form a function
		},
		{
			// Compute the Jacobian matrix
			df_val = ~df.value(x);
			"Jacobian at step %: \n%".format(step, df_val).postln;  // Print the Jacobian matrix for analysis

			// Update temporal solution using shiftL and small random noise
			temporal_solution = (~shiftL.value(x) + RedVector2D.rand(n, -0.99, 0.99)).asArray;  // Adding small random noise and converting to array
			"temporal_solution at step %: %".format(step, temporal_solution).postln;

			// Calculate new function value and dot product
			f_temp = ~a.value(temporal_solution.asArray);
			dot_f_temp = ~dotProduct.value(f_temp, f_temp);
			"dot_f_temp at step %: %".format(step, dot_f_temp).postln;

			// If the new solution improves, update the best solution
			if (dot_f_temp < dot_f_init) {
				solution = temporal_solution.asArray;  // Ensure solution is an array
				dot_f_init = dot_f_temp;
				"Updated solution at step %: %".format(step, solution).postln;
			};

			// Stopping criterion based on improvement tolerance
			if (dot_f_init < tolerance) {
				"Convergence achieved at step % with tolerance.".format(step).postln;
				escape = true;  // Stop further iterations if tolerance is met
			};

			// Increment step counter
			step = step + 1;
		});

		// Sort the solution from highest to lowest
		solution = solution.sort.reverse;

		// Return the final sorted solution as an array
		"Final sorted solution: %".format(solution).postln;
		solution.as(Array);  // Ensure the returned result is an array
	};

	//x = 1.0.rand!16;
	//~min_func.value(x);
})
);

////////////////////////////////////////////////////////////////////////////////
//synthesis application
//generate carrier complex amplitudes
//sometimes it returns 0s for all amplitudes, in that case run it again
(
~numOfIterations = 100; //higher number = longer
~numberOfParrtials = 16;
//generate a random set of amplitudes
~targetAmplitudes = 1.0.rand!~numberOfParrtials;
~carrierComplexAmplitudes = ~min_func.value(x: ~targetAmplitudes, maxIterations: 500);
)

(

Ndef(\sineCluster, {|carrierPitch = 1280, targetPitch = 130|

	var size,freqCluster, phaseAlign, sines;

	size = ~targetAmplitudes.size;


	freqCluster = (size-1).collect{|i|

		//i.postln;
		carrierPitch + ((i + 1) * targetPitch);

	};

	freqCluster = freqCluster.addFirst(carrierPitch);

	//null signal
	phaseAlign = SinOscOS.ar(carrierPitch, 0, 0, 0);

	sines = size.collect{|i|

		var sine = SinOscOS.ar(
			freq: freqCluster[i],
			phase: phaseAlign,
			oversample: 4,
			mul: ~carrierComplexAmplitudes[i]);

		sine * 0.2;
	};

	sines.sum!2;

});
);

Ndef(\sineCluster).play;

//reset parameters' values
Ndef(\sineCluster).set(\carrierPitch, 2202);
Ndef(\sineCluster).set(\carrierPitch, 912);
Ndef(\sineCluster).set(\targetPitch, 229);
Ndef(\sineCluster).set(\targetPitch, 72);

(
Ndef(\sineCluster).play;
//sequence parameters in time
Tdef(\playSeq, {

	var acc = 2.5;
	var targetPitchSeq = Pseq([72, 80, 86, 192], inf).asStream;
	var carrierSeq = Pseq([2272, 2680, 2004, 2457], inf).asStream;
	var time = Pseq([0.2, 0.8, 0.4, 0.2], inf).asStream * acc.reciprocal;

	loop{
		Ndef(\sineCluster).set(\targetPitch, targetPitchSeq.next + 15.rand);
		Ndef(\sineCluster).set(\carrierPitch, carrierSeq.next + 123.rand);
		//time
		(time.next).wait;
	}
}).play;
)

Ndef(\sineCluster).stop;


//modulated/pattern carrier and target
(

Ndef(\sineCluster, {|modfreq = 10.94, carrierPitch = 2370, targetPitch = 120|

	var env, rate, carrierMod, targetMod, size,freqCluster, sines;

	//rate = LFNoise2.ar(0.02).exprange(6, 14.03);
	rate = 4.7;
	//modulators
	carrierMod = carrierPitch * Latch.kr(
		LFSaw.kr(modfreq, mul: modfreq/29, add: modfreq/29),
		Impulse.kr(rate)
	).linlin(-1, 1, 0.1, 2.0);

	targetMod = targetPitch * Latch.kr(
		LFSaw.kr(modfreq, mul: modfreq/29, add: modfreq/29),
		Impulse.kr(rate)
	).linlin(-1, 1, 0.1, 1.5);

	size = ~targetAmplitudes.size;

	freqCluster = (size-1).collect{|i|

		//i.postln;
		carrierMod + ((i + 1) * targetMod);

	};

	freqCluster = freqCluster.addFirst(carrierPitch * Latch.kr(
		LFSaw.kr(modfreq, mul: modfreq/49, add: modfreq/214),
		Impulse.kr(rate)
	).linlin(-1, 1, 0.5, 2));

	sines = size.collect{|i|

		var sine = SinOscOS.ar(freqCluster[i], oversample: 2, mul: ~carrierComplexAmplitudes[i]);
		var delay = DelayC.ar(sine, 0.001 + (0.001 * i), 0.001 + (0.001 * i));
		env = Env.new(
			levels: [0, 0.5, 0.0, 0.0, 0.9, 0.5, 0.5, 0.0],
			times: [0.0001, 1.3/rate, 0.8, 0.0, 1.5/rate, 0.2, 0.2, 0.1],
			curve: Demand.kr(Impulse.kr(rate), 0, Dseq([-6, 9], inf)),
			releaseNode: nil,
			loopNode: nil,
			offset: 0.1
		);

		env = EnvGen.kr(env, gate: Impulse.kr(rate));
		delay * env

		//sine * 0.2;
	};




	Pan2.ar(sines.sum * env, Demand.kr(Impulse.kr(rate), 0, Dseq([1, -1, 1, 1, -1], inf)));


});
);

Ndef(\sineCluster).play;
//reset parameters
Ndef(\sineCluster).set(\modfreq, 11.94);
Ndef(\sineCluster).set(\carrierPitch, 1213.2);
Ndef(\sineCluster).set(\targetPitch, 39.2);

//get server window
s.makeWindow;



