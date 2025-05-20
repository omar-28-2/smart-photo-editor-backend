from flask_restx import Namespace, Resource

mask_ns = Namespace('mask', description='Mask operations')

@mask_ns.route('/apply')
class ApplyMask(Resource):
    def post(self):
        
        return {"message": "Mask applied (placeholder)"}
