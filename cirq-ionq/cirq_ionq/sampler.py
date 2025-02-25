# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A `cirq.Sampler` implementation for the IonQ API."""

from typing import List, Optional, TYPE_CHECKING

from cirq_ionq import results
import cirq

if TYPE_CHECKING:
    import cirq_ionq


class Sampler(cirq.Sampler):
    """A sampler that works against the IonQ API.

    Users should get a sampler from the `sampler` method on `cirq_ionq.Service`.

    Example of using this sampler:
            >> service = cirq_ionq.Service(...)
            >> a, b, c = cirq.LineQubit.range(3)
            >> sampler = service.sampler()
            >> circuit = cirq.Circuit(cirq.X(a), cirq.measure(a, key='out'))
            >> print(sampler.sample(circuit, repetitions=4))
               out
            0    1
            1    1
            2    1
            3    1
    """

    def __init__(
        self,
        service: 'cirq_ionq.Service',
        target: Optional[str],
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ):
        """Construct the sampler.

        Users should get a sampler from the `sampler` method on `cirq_ionq.Service`.

        Args:
            service: The service used to create this sample.
            target: Where to run the job. Can be 'qpu' or 'simulator'. If this is not specified,
                there must be a default target set on `service`.
            seed: If the target is `simulation` the seed for generating results. If None, this
                will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise
                must be a modulate similar to `np.random`.
        """
        self._service = service
        self._target = target
        self._seed = seed

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> List['cirq.Result']:
        """Runs a sweep for the given Circuit.

        Note that this creates jobs for each of the sweeps in the given sweepable, and then
        blocks until all of the jobs are complete.

        See `cirq.Sampler` for documentation on args.

        For use of the `sample` method, see the documentation of `cirq.Sampler`.
        """
        resolvers = [r for r in cirq.to_resolvers(params)]
        jobs = [
            self._service.create_job(
                circuit=cirq.resolve_parameters(program, resolver),
                repetitions=repetitions,
                target=self._target,
            )
            for resolver in resolvers
        ]
        job_results = [job.results() for job in jobs]
        cirq_results = []
        for result, params in zip(job_results, resolvers):
            if isinstance(result, results.QPUResult):
                cirq_results.append(result.to_cirq_result(params=params))
            else:
                cirq_results.append(result.to_cirq_result(params=params, seed=self._seed))
        return cirq_results
