from flask_restful import Resource
import tensorflow as tf


class USE(Resource):
    def post(self):

        return (
            {
                "vector": tf.test.is_gpu_available(
                    cuda_only=False, min_cuda_compute_capability=None
                )
            },
            201,
        )

