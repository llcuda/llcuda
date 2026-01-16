#!/usr/bin/env python3
"""
llcuda v2.2.0 Module Integration Test Script
=============================================

This script tests the llcuda Python module APIs on Kaggle 2× T4.
Run separately after the main build notebook completes.

Usage in Kaggle:
    !python test_llcuda_modules.py

Requirements:
    - llcuda v2.2.0 installed: pip install git+https://github.com/llcuda/llcuda.git
    - RAPIDS available (cudf, cugraph)
    - Kaggle 2× T4 GPUs
"""

import os
import sys

def test_basic_imports():
    """Test 1: Basic llcuda imports"""
    print("=" * 70)
    print("TEST 1: Basic llcuda Imports")
    print("=" * 70)
    
    try:
        import llcuda
        print(f"✅ llcuda version: {llcuda.__version__}")
        print(f"✅ Available exports: {len(llcuda.__all__)}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_split_gpu_config():
    """Test 2: SplitGPUConfig class"""
    print("\n" + "=" * 70)
    print("TEST 2: SplitGPUConfig")
    print("=" * 70)
    
    try:
        import llcuda
        
        config = llcuda.SplitGPUConfig(llm_gpu=0, graph_gpu=1)
        print(f"✅ LLM GPU: {config.llm_gpu}")
        print(f"✅ Graph GPU: {config.graph_gpu}")
        print(f"✅ LLM env: {config.llm_env()}")
        print(f"✅ Graph env: {config.graph_env()}")
        
        # Test llama_server_cmd with correct API
        cmd = config.llama_server_cmd(
            model_path="/path/to/model.gguf",
            port=8080
        )
        print(f"✅ Server command: {cmd[:60]}...")
        
        return True
    except Exception as e:
        print(f"❌ SplitGPUConfig failed: {e}")
        return False


def test_graphistry_module():
    """Test 3: Graphistry module (if available)"""
    print("\n" + "=" * 70)
    print("TEST 3: Graphistry Module")
    print("=" * 70)
    
    try:
        from llcuda import graphistry
        print(f"✅ Graphistry module imported")
        
        # Check what's available
        exports = [x for x in dir(graphistry) if not x.startswith('_')]
        print(f"✅ Graphistry exports: {exports}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Graphistry module not available: {e}")
        print("   This is expected if graphistry is a placeholder module")
        return False
    except Exception as e:
        print(f"❌ Graphistry failed: {e}")
        return False


def test_louie_module():
    """Test 4: Louie module (if available)"""
    print("\n" + "=" * 70)
    print("TEST 4: Louie Module")
    print("=" * 70)
    
    try:
        from llcuda import louie
        print(f"✅ Louie module imported")
        
        # Check what's available
        exports = [x for x in dir(louie) if not x.startswith('_')]
        print(f"✅ Louie exports: {exports}")
        
        return True
    except ImportError as e:
        print(f"⚠️  Louie module not available: {e}")
        print("   This is expected if louie is a placeholder module")
        return False
    except Exception as e:
        print(f"❌ Louie failed: {e}")
        return False


def test_rapids_backend():
    """Test 5: RAPIDS/cuGraph backend"""
    print("\n" + "=" * 70)
    print("TEST 5: RAPIDS Backend (Direct)")
    print("=" * 70)
    
    try:
        import cudf
        import cugraph
        print(f"✅ cuDF version: {cudf.__version__}")
        print(f"✅ cuGraph version: {cugraph.__version__}")
        
        # Create sample graph
        edges = cudf.DataFrame({
            "src": [0, 1, 2, 3, 4],
            "dst": [1, 2, 3, 4, 0],
            "weight": [1.0, 2.0, 1.5, 0.5, 3.0]
        })
        
        G = cugraph.Graph()
        G.from_cudf_edgelist(edges, source="src", destination="dst")
        print(f"✅ Graph created: {G.number_of_vertices()} vertices, {G.number_of_edges()} edges")
        
        # PageRank
        pagerank = cugraph.pagerank(G)
        top_node = pagerank.nlargest(1, 'pagerank')['vertex'].values[0]
        print(f"✅ PageRank top node: {top_node}")
        
        return True
    except ImportError as e:
        print(f"⚠️  RAPIDS not available: {e}")
        return False
    except Exception as e:
        print(f"❌ RAPIDS test failed: {e}")
        return False


def test_server_manager():
    """Test 6: ServerManager class"""
    print("\n" + "=" * 70)
    print("TEST 6: ServerManager")
    print("=" * 70)
    
    try:
        from llcuda import ServerManager
        print(f"✅ ServerManager imported")
        
        # Just test instantiation, don't actually start server
        sm = ServerManager.__doc__
        print(f"✅ ServerManager docstring exists: {bool(sm)}")
        
        return True
    except Exception as e:
        print(f"❌ ServerManager failed: {e}")
        return False


def test_inference_engine():
    """Test 7: InferenceEngine class"""
    print("\n" + "=" * 70)
    print("TEST 7: InferenceEngine")
    print("=" * 70)
    
    try:
        from llcuda import InferenceEngine
        print(f"✅ InferenceEngine imported")
        
        return True
    except Exception as e:
        print(f"❌ InferenceEngine failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("llcuda v2.2.0 MODULE INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    results = {
        "basic_imports": test_basic_imports(),
        "split_gpu_config": test_split_gpu_config(),
        "graphistry": test_graphistry_module(),
        "louie": test_louie_module(),
        "rapids": test_rapids_backend(),
        "server_manager": test_server_manager(),
        "inference_engine": test_inference_engine(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
