#!/usr/bin/env python3
"""
Tests for directory helper functions in cartpole.py
"""
import pytest
import os
import glob
from datetime import datetime
import time

# Import the functions we're testing
from src.cartpole import get_run_directory, ensure_directory_exists, resolve_checkpoint_path


class TestGetRunDirectory:
    """Test the get_run_directory function."""
    
    def test_default_single_run(self):
        """Test default single run directory generation."""
        run_dir = get_run_directory()
        assert run_dir.startswith('runs/single/train_')
        assert len(run_dir.split('/')) == 3
        
    def test_custom_run_id(self):
        """Test with custom run ID."""
        run_dir = get_run_directory(run_id='my_custom_run')
        assert run_dir == 'runs/single/my_custom_run'
        
    def test_search_run_type(self):
        """Test search run type."""
        run_dir = get_run_directory(run_type='search')
        assert run_dir.startswith('runs/search/train_')
        
    def test_custom_run_type_and_id(self):
        """Test custom run type and ID."""
        run_dir = get_run_directory(run_type='custom', run_id='test123')
        assert run_dir == 'runs/custom/test123'
        
    def test_timestamp_format(self):
        """Test that timestamp follows expected format."""
        run_dir = get_run_directory()
        timestamp_part = run_dir.split('/')[-1].replace('train_', '')
        # Should be YYYYMMDD_HHMMSS format
        assert len(timestamp_part) == 15
        assert timestamp_part[8] == '_'
        # Verify it's a valid timestamp
        datetime.strptime(timestamp_part, '%Y%m%d_%H%M%S')


class TestEnsureDirectoryExists:
    """Test the ensure_directory_exists function."""
    
    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        test_dir = tmp_path / "new_test_dir"
        assert not test_dir.exists()
        
        result = ensure_directory_exists(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == str(test_dir)
        
    def test_existing_directory(self, tmp_path):
        """Test with an existing directory."""
        test_dir = tmp_path / "existing_dir"
        test_dir.mkdir()
        
        result = ensure_directory_exists(str(test_dir))
        assert test_dir.exists()
        assert result == str(test_dir)
        
    def test_nested_directory_creation(self, tmp_path):
        """Test creating nested directories."""
        test_dir = tmp_path / "level1" / "level2" / "level3"
        assert not test_dir.exists()
        
        result = ensure_directory_exists(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == str(test_dir)


class TestResolveCheckpointPath:
    """Test the resolve_checkpoint_path function."""
    
    def test_existing_file_as_is(self, tmp_path):
        """Test with a file that exists at the given path."""
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_text("dummy")
        
        result = resolve_checkpoint_path(str(checkpoint))
        assert result == str(checkpoint)
        
    def test_file_in_runs_single(self, tmp_path):
        """Test finding file in runs/single/* directory."""
        # Create directory structure
        runs_dir = tmp_path / "runs" / "single" / "train_123"
        runs_dir.mkdir(parents=True)
        checkpoint = runs_dir / "model.pt"
        checkpoint.write_text("dummy")
        
        # Change to tmp_path to simulate being in project root
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = resolve_checkpoint_path("model.pt")
            # Should return relative path as found by glob
            assert result == "runs/single/train_123/model.pt"
        finally:
            os.chdir(original_cwd)
            
    def test_file_in_runs_search(self, tmp_path):
        """Test finding file in runs/search/*/* directory."""
        # Create directory structure
        search_dir = tmp_path / "runs" / "search" / "random_123" / "trial_1"
        search_dir.mkdir(parents=True)
        checkpoint = search_dir / "model.pt"
        checkpoint.write_text("dummy")
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = resolve_checkpoint_path("model.pt")
            # Should return relative path as found by glob
            assert result == "runs/search/random_123/trial_1/model.pt"
        finally:
            os.chdir(original_cwd)
            
    def test_multiple_matches_returns_newest(self, tmp_path):
        """Test that multiple matches return the most recent file."""
        # Create multiple checkpoints
        runs_dir = tmp_path / "runs" / "single"
        runs_dir.mkdir(parents=True)
        
        # Create older checkpoint
        old_dir = runs_dir / "train_old"
        old_dir.mkdir()
        old_checkpoint = old_dir / "model.pt"
        old_checkpoint.write_text("old")
        
        # Wait a bit to ensure different mtime
        time.sleep(0.1)
        
        # Create newer checkpoint
        new_dir = runs_dir / "train_new"
        new_dir.mkdir()
        new_checkpoint = new_dir / "model.pt"
        new_checkpoint.write_text("new")
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = resolve_checkpoint_path("model.pt")
            # Should return the newer file's relative path
            assert result == "runs/single/train_new/model.pt"
        finally:
            os.chdir(original_cwd)
            
    def test_no_matches_returns_original(self, tmp_path):
        """Test that original path is returned when no matches found."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = resolve_checkpoint_path("nonexistent.pt")
            assert result == "nonexistent.pt"
        finally:
            os.chdir(original_cwd)
            
    def test_relative_path_in_runs(self, tmp_path):
        """Test finding file with relative path in runs directory."""
        # Create directory structure
        runs_dir = tmp_path / "runs" / "single" / "train_123"
        runs_dir.mkdir(parents=True)
        subdir = runs_dir / "checkpoints"
        subdir.mkdir()
        checkpoint = subdir / "model.pt"
        checkpoint.write_text("dummy")
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            result = resolve_checkpoint_path("checkpoints/model.pt")
            # Should return relative path as found by glob
            assert result == "runs/single/train_123/checkpoints/model.pt"
        finally:
            os.chdir(original_cwd)


class TestDirectoryIntegration:
    """Integration tests for directory functions working together."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete directory workflow."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create a run directory
            run_dir = get_run_directory(run_id='test_workflow')
            ensure_directory_exists(run_dir)
            
            # Verify it was created
            assert os.path.exists(run_dir)
            assert os.path.isdir(run_dir)
            
            # Create a checkpoint in it
            checkpoint_path = os.path.join(run_dir, 'model.pt')
            with open(checkpoint_path, 'w') as f:
                f.write('test')
            
            # Verify we can resolve it
            resolved = resolve_checkpoint_path('model.pt')
            assert resolved == checkpoint_path
            
        finally:
            os.chdir(original_cwd)
            
    def test_multiple_run_types(self, tmp_path):
        """Test creating directories for different run types."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # Create single run
            single_dir = get_run_directory(run_type='single', run_id='single_test')
            ensure_directory_exists(single_dir)
            
            # Create search run
            search_dir = get_run_directory(run_type='search', run_id='search_test')
            ensure_directory_exists(search_dir)
            
            # Verify structure
            assert os.path.exists('runs/single/single_test')
            assert os.path.exists('runs/search/search_test')
            
        finally:
            os.chdir(original_cwd)