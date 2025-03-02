'use client';

import { Box, Button, Checkbox, FileInput, Image, Input, LoadingOverlay, SimpleGrid, Stack, Switch, Textarea } from "@mantine/core";
import { useEffect, useRef, useState } from "react";
import { BarChart } from '@mantine/charts';

export default function Home() {
  const [image, setImage] = useState('https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/images/bg-7.png');
  const [classCount, setClassCount] = useState<{ key: string; value: unknown }[]>([]);
  const [visible, setVisible] = useState(false);
  const [answer, setAnswer] = useState('');

  const question = 'Whatever';
  const answer_ = `Pros:\nImproved Transportation Convenience: The new runway can alleviate flight congestion, enhancing airport capacity and flight efficiency.\nEconomic Growth: It will boost employment, business, and tourism, increasing regional economic vitality.\nEnhanced International Connectivity: Strengthening Heathrow Airport’s global competitiveness, attracting more flights and commercial activities.\n\nCons:\nEnvironmental Impact: Noise pollution, air pollution, and ecological damage, especially to green spaces and waterways.\nSocial Impact: Nearby residents may be disturbed by noise and pollution, possibly requiring relocation, leading to changes in community structure.\nHigh Costs: The runway construction requires significant investment, which could put pressure on local finances.\n\n Mitigation Strategies:\nNoise Control: Optimize flight paths and install noise barriers.\nEnvironmental Protection: Strengthen ecological restoration and reduce emissions.\nCommunity Compensation: Provide relocation support or resettlement options to reduce social conflicts.`;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const file = formData.get('image') as File;

    if (file) {
      try {
        const reader = new FileReader();
        reader.onloadend = () => {
          setImage(reader.result as string);
        };
        reader.readAsDataURL(file);

        setVisible(true);
        const response = await fetch('http://localhost:8000/statistic', {
          method: 'POST',
          body: formData,
        });

        setVisible(false);
        if (!response.ok) {
          throw new Error('File upload failed');
        } else {
          const data = await response.json();
          console.log(data);
          setClassCount(Object.entries(data).map(([key, value]) => ({ key, value })));
          console.log(classCount);
        }
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
      <div className="m-3">
        <SimpleGrid cols={2}>
          <div style={{ overflow: 'hidden', borderRadius: '500px' }}>
            <Box style={{
              position: 'relative',
            }}>
              <Image
                src={image}
                alt="Background"
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                }}
              />
            </Box>
          </div>
          <div>
            <LoadingOverlay visible={visible} zIndex={1000} overlayProps={{ radius: "sm", blur: 2 }} />
            <BarChart
              data={classCount}
              color="blue"
              style={{ height: 300 }} dataKey={"key"} series={[{ name: 'value', color: 'violet.6' }]} />
          </div>
        </SimpleGrid>
      </div>
      <div className="m-3">
        <Input defaultValue={question}/>
        <Button onClick={() => setAnswer(answer_)} m={3}>Ask</Button>
        <Textarea placeholder="Answer" m={3} value={answer} readOnly rows={20} />
      </div>
    </>
  );
}
