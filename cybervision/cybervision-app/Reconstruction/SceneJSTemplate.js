var myNode = {
  "type": "node",
  "nodes": [
	{
	  "type": "node",
	  "id": "myModel.scene",
	  "nodes": [
		{
		  "type": "node",
		  "id": "myModel.VisualSceneNode",
		  "nodes": [
			{
			  "type": "node",
			  "id": "myModel.Box",
			  "nodes": [
				{
				  "type": "rotate",
				  "sid": "rotateX",
				  "x": 1,
				  "y": 0,
				  "z": 0,
				  "angle": 0,
				  "nodes": [
					{
					  "type": "rotate",
					  "sid": "rotateY",
					  "x": 0,
					  "y": 1,
					  "z": 0,
					  "angle": 0,
					  "nodes": [
						{
						  "type": "rotate",
						  "sid": "rotateZ",
						  "x": 0,
						  "y": 0,
						  "z": 1,
						  "angle": 0,
						  "nodes": [
							{
							  "type": "node",
							  "id": "myModel.box-lib",
							  "nodes": [
								{
								  "type": "material",
								  "id": "myModel.Blue-fx",
								  "baseColor": {
									"r": 1,
									"g": 1,
									"b": 1
								  },
								  "specularColor": {
									"r": 1,
									"g": 1,
									"b": 1,
									"a": 1
								  },
								  "shine": 50,
								  "specular": 1,
								  "nodes": [
									{
									  "type": "texture",
									  "sid": "texture",
									  "layers": [
										{
										  "uri": "./##[texture-image-filename]##",
										  "applyTo": "baseColor",
										  "flipY": false,
										  "blendMode": "add",
										  "wrapS": "repeat",
										  "wrapT": "repeat",
										  "minFilter": "linearMipMapLinear",
										  "magFilter": "linear"
										}
									  ],
									  "nodes": [
										{
										  "type": "geometry",
										  "primitive": "triangles",
										  "positions": [##[points]##],
										  "normals": [##[normals]##],
										  "uv": [##[texture-coordinates]##],
										  "indices": [##[triangles-indexes]##]
										}
									  ]
									}
								  ]
								}
							  ]
							}
						  ]
						}
					  ]
					}
				  ]
				}
			  ]
			}
		  ]
		}
	  ]
	}
  ]
};
