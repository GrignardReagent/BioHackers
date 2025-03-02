'use client';

import { Box, Button, Checkbox, FileInput, Image, SimpleGrid, Stack, Switch } from "@mantine/core";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const file = formData.get('image') as File;

    if (file) {
      try {
        const response = await fetch('http://localhost:8000/statistic', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error('File upload failed');
        }

        alert('File uploaded successfully');
      } catch (error) {
        console.error('Error uploading file:', error);
        alert('Error uploading file');
      }
    } else {
      alert('Please select a file to upload');
    }
  };

  return (
    <>
    <form className="m-3" onSubmit={handleSubmit}>
            <FileInput
              m={3}
              placeholder="Upload image"
              name="image"
            />
            <Button m={3} type="submit">Upload</Button>
          </form>
    </>
  );
}
